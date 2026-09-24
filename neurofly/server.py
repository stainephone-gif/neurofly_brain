"""Live simulation server for the gallery screen.

One process, one port: it serves the viewer (``web/``), the neuron atlas
and meshes over plain HTTP, and streams spikes over a WebSocket while
accepting commands from any number of connected screens.

    neurofly serve --port 8765
    # then open http://localhost:8765

Protocol (WebSocket):

client -> server, JSON text::

    {"cmd": "activate",   "idx": [..], "rate": 100, "expand": "type"|null}
    {"cmd": "deactivate", "idx": [..]}          # no idx: all
    {"cmd": "silence",    "idx": [..], "expand": ...}
    {"cmd": "unsilence",  "idx": [..]}          # no idx: all
    {"cmd": "connect",    "pre": i, "post": j, "n": 10, "sign": 1}
    {"cmd": "disconnect", "pre": i, "post": j}  # both optional
    {"cmd": "reset"} | {"cmd": "clear"} | {"cmd": "pause"} | {"cmd": "play"}
    {"cmd": "speed", "value": 1.0}              # target bio-s per wall-s
    {"cmd": "info", "idx": i}                   # ask about one neuron
    {"cmd": "type", "idx": i}                   # members of its cell type

server -> client:

* binary frames, one per simulation window: ``float32 t_ms, float32
  window_ms, uint32 n`` followed by ``n`` ``uint32`` indices of the neurons
  that spiked (a neuron appears once per spike);
* JSON text ``{"type": "status", ...}`` twice a second with the current
  manipulations, rates and timing, and ``{"type": "info", ...}`` /
  ``{"type": "type", ...}`` as replies.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import struct
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from . import neurons as N
from .annotations import Atlas
from .archive import NEUROPILS, load_mesh, skeleton_segments
from .data import load_brain
from .model import FlyBrain

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# ------------------------------------------------------------------ presets
def build_presets(brain: FlyBrain, atlas: Atlas) -> list[dict]:
    """Named neuron groups shown as buttons on the screen."""
    def idx(ids):
        return [int(i) for i in brain.index(ids)]

    presets = [
        {"key": "sugar", "label": "Вкус сахара", "hint": "21 вкусовой нейрон → хоботок (MN9)", "idx": idx(N.named("sugar", brain.release)), "rate": 200},
        {"key": "p9", "label": "DNp09: вперёд", "hint": "нисходящие нейроны, команда идти вперёд", "idx": idx(N.P9), "rate": 100},
    ]
    for key, label, hint, rate in [
        ("MDN", "MDN: назад", "нисходящие нейроны, задний ход", 100),
        ("DNa02", "DNa02: поворот", "нисходящие нейроны поворота", 100),
        ("DNp01", "Гигантское волокно", "рефлекс прыжка от угрозы", 100),
    ]:
        members = atlas.by_type(key)
        if members.size:
            presets.append({"key": key, "label": label, "hint": hint, "idx": [int(i) for i in members], "rate": rate})
    presets.append({"key": "mn9", "label": "MN9: хоботок", "hint": "мотонейрон хоботка, наблюдаемый выход", "idx": idx([N.MN9]), "rate": 0})
    return presets


# --------------------------------------------------------------- simulation
class Simulation(threading.Thread):
    """Runs the brain in its own thread; talks to asyncio through queues."""

    def __init__(self, brain: FlyBrain, atlas: Atlas, window_ms: float = 10.0):
        super().__init__(daemon=True)
        self.brain = brain
        self.atlas = atlas
        self.window_ms = window_ms
        self.commands: Queue = Queue()
        self.frames: Queue = Queue(maxsize=64)
        self.paused = False
        self.speed = 1.0          # target bio seconds per wall second
        self.realtime_ratio = 0.0  # measured bio / wall
        self.spikes_per_s = 0.0
        self.active_recent = 0
        self.lock = threading.Lock()
        self.window_steps = brain.p.steps(window_ms)

    def run(self) -> None:
        b = self.brain
        ema_wall = None
        recent = []
        while True:
            self._drain_commands()
            if self.paused:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            with self.lock:
                spk = []
                for _ in range(self.window_steps):
                    s = b.step()
                    if s.size:
                        spk.append(s)
                t_ms = b.t_ms
            idx = np.concatenate(spk).astype(np.uint32) if spk else np.empty(0, np.uint32)
            wall = time.perf_counter() - t0
            ema_wall = wall if ema_wall is None else 0.9 * ema_wall + 0.1 * wall
            self.realtime_ratio = (self.window_ms / 1000) / max(ema_wall, 1e-9)
            recent.append(idx)
            if len(recent) > int(1000 / self.window_ms):
                recent.pop(0)
            allr = np.concatenate(recent) if recent else idx
            self.spikes_per_s = len(allr) / (len(recent) * self.window_ms / 1000)
            self.active_recent = int(np.unique(allr).size)
            frame = struct.pack("<ffI", t_ms, self.window_ms, idx.size) + idx.tobytes()
            try:
                self.frames.put_nowait(frame)
            except Exception:
                pass  # viewer is slow: drop the frame
            # pacing when the machine is faster than the requested speed
            budget = (self.window_ms / 1000) / self.speed
            if wall < budget:
                time.sleep(budget - wall)

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd, reply = self.commands.get_nowait()
            except Empty:
                return
            try:
                out = self._apply(cmd)
            except Exception as e:  # bad ids etc. must not kill the loop
                out = {"type": "error", "message": str(e)}
            if reply is not None:
                reply(out)

    def _expand(self, idx, how):
        idx = [int(i) for i in idx]
        if not how or not idx:
            return idx
        df = self.atlas.df
        out = set()
        for i in idx:
            row = df.iloc[i]
            ct = str(row.cell_type) or str(row.hemibrain_type)
            if not ct:
                out.add(i)
                continue
            side = str(row.side) if how == "type_side" else None
            out.update(int(k) for k in self.atlas.by_type(ct, side))
        return sorted(out)

    def _apply(self, cmd: dict):
        b = self.brain
        c = cmd.get("cmd")
        with self.lock:
            if c == "activate":
                idx = self._expand(cmd.get("idx", []), cmd.get("expand"))
                b.activate(idx, float(cmd.get("rate", 100)))
                return {"type": "ack", "cmd": c, "n": len(idx)}
            if c == "deactivate":
                idx = cmd.get("idx")
                b.deactivate(self._expand(idx, cmd.get("expand")) if idx else None)
            elif c == "silence":
                idx = self._expand(cmd.get("idx", []), cmd.get("expand"))
                b.silence(idx)
                return {"type": "ack", "cmd": c, "n": len(idx)}
            elif c == "unsilence":
                idx = cmd.get("idx")
                b.unsilence(self._expand(idx, cmd.get("expand")) if idx else None)
            elif c == "connect":
                b.connect(int(cmd["pre"]), int(cmd["post"]), float(cmd.get("n", 10)), int(cmd.get("sign", 1)))
            elif c == "disconnect":
                pre, post = cmd.get("pre"), cmd.get("post")
                b.disconnect(None if pre is None else int(pre), None if post is None else int(post))
            elif c == "reset":
                b.reset()
            elif c == "clear":
                b.clear()
            elif c == "pause":
                self.paused = True
            elif c == "play":
                self.paused = False
            elif c == "speed":
                self.speed = max(0.01, float(cmd.get("value", 1.0)))
            elif c == "info":
                return self._info(int(cmd["idx"]))
            elif c == "type":
                i = int(cmd["idx"])
                return {"type": "type", "idx": i, "members": self._expand([i], "type")}
            else:
                return {"type": "error", "message": f"unknown command {c!r}"}
        return {"type": "ack", "cmd": c}

    def _info(self, i: int) -> dict:
        b, df = self.brain, self.atlas.df
        row = df.iloc[i]
        out_idx, out_w = b.synapses_of(i, "out")
        in_idx, in_w = b.synapses_of(i, "in")
        order = np.argsort(-np.abs(out_w))[:200]
        order_in = np.argsort(-np.abs(in_w))[:200]
        return {
            "type": "info",
            "idx": i,
            "id": str(int(b.ids[i])),
            "label": self.atlas.describe(i),
            "cell_type": str(row.cell_type) or str(row.hemibrain_type),
            "super_class": str(row.super_class),
            "cell_class": str(row.cell_class),
            "side": str(row.side),
            "nt": str(row.nt),
            "n_out": int(out_idx.size),
            "n_in": int(in_idx.size),
            "syn_out": int(np.abs(out_w).sum()),
            "syn_in": int(np.abs(in_w).sum()),
            "out": [[int(out_idx[k]), float(out_w[k])] for k in order],
            "in": [[int(in_idx[k]), float(in_w[k])] for k in order_in],
            "stim": float(b.stim_rate[i]),
            "silenced": bool(b.silenced[i]),
        }

    def status(self) -> dict:
        b = self.brain
        with self.lock:
            stim = np.flatnonzero(b.stim_rate)
            return {
                "type": "status",
                "t_ms": b.t_ms,
                "paused": self.paused,
                "speed": self.speed,
                "realtime": self.realtime_ratio,
                "spikes_per_s": self.spikes_per_s,
                "active": self.active_recent,
                "stim": [[int(i), float(b.stim_rate[i])] for i in stim[:5000]],
                "n_stim": int(stim.size),
                "silenced": [int(i) for i in np.flatnonzero(b.silenced)[:5000]],
                "n_silenced": int(b.silenced.sum()),
                "extra": [[int(pre), int(p), float(w)] for pre, (posts, ws) in b.extra.items() for p, w in zip(posts, ws)][:5000],
                "n_extra": b.n_extra(),
            }


# ------------------------------------------------------------------- server
class GalleryServer:
    def __init__(self, brain: FlyBrain, atlas: Atlas, window_ms: float = 10.0):
        self.brain = brain
        self.atlas = atlas
        self.sim = Simulation(brain, atlas, window_ms)
        self.clients: set = set()
        self.presets = build_presets(brain, atlas)
        self._neurons_bin = None
        self._meta = None
        self._mesh_cache: dict[str, bytes] = {}

    # ---- static payloads
    def neurons_bin(self) -> bytes:
        if self._neurons_bin is None:
            df = self.atlas.df
            pos = self.atlas.positions().astype("<f4")
            codes = []
            for col in ["super_class", "side", "nt"]:
                cat = df[col].astype("category")
                codes.append(cat.cat.codes.to_numpy().astype(np.uint8))
            self._neurons_bin = pos.tobytes() + b"".join(c.tobytes() for c in codes)
        return self._neurons_bin

    def meta(self) -> dict:
        if self._meta is None:
            df = self.atlas.df
            pos = self.atlas.positions()
            self._meta = {
                "n": int(self.brain.n),
                "release": self.brain.release,
                "super_classes": list(map(str, df["super_class"].astype("category").cat.categories)),
                "sides": list(map(str, df["side"].astype("category").cat.categories)),
                "nts": list(map(str, df["nt"].astype("category").cat.categories)),
                "bounds": [pos.min(0).tolist(), pos.max(0).tolist()],
                "presets": self.presets,
                "neuropils": NEUROPILS,
                "window_ms": self.sim.window_ms,
                "n_connections": int(self.brain.W.nnz),
                "n_synapses": int(np.abs(self.brain.W.data).sum()),
            }
        return self._meta

    def mesh_bin(self, name: str) -> bytes:
        if name not in self._mesh_cache:
            m = load_mesh(name)
            v = m["vertices"].astype("<f4")
            f = m["faces"].astype("<u4")
            self._mesh_cache[name] = struct.pack("<II", len(v), len(f)) + v.tobytes() + f.tobytes()
        return self._mesh_cache[name]

    def skeleton_bin(self, idx: int) -> bytes:
        seg = skeleton_segments(int(self.brain.ids[idx])).astype("<f4")
        return struct.pack("<I", len(seg)) + seg.tobytes()

    # ---- HTTP
    def http(self, connection, request):
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        path = request.path.split("?")[0]
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None

        def resp(body: bytes, ctype: str, status=200):
            h = Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                         ("Cache-Control", "no-cache"), ("Access-Control-Allow-Origin", "*")])
            return Response(status, {200: "OK", 204: "No Content", 404: "Not Found"}.get(status, "Error"), h, body)

        try:
            if path == "/favicon.ico":
                return resp(b"", "image/x-icon", 204)
            if path in ("/", "/index.html"):
                return resp((WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/meta.json":
                return resp(json.dumps(self.meta()).encode(), "application/json")
            if path == "/api/neurons.bin":
                return resp(self.neurons_bin(), "application/octet-stream")
            if path.startswith("/api/mesh/"):
                return resp(self.mesh_bin(path.rsplit("/", 1)[1]), "application/octet-stream")
            if path.startswith("/api/skeleton/"):
                return resp(self.skeleton_bin(int(path.rsplit("/", 1)[1])), "application/octet-stream")
            if path.startswith("/api/search"):
                q = request.path.split("q=", 1)[1] if "q=" in request.path else ""
                from urllib.parse import unquote
                hits = self.atlas.search(unquote(q), limit=30)
                return resp(hits.to_json(orient="records").encode(), "application/json")
            local = (WEB_DIR / path.lstrip("/")).resolve()
            if WEB_DIR in local.parents and local.is_file():
                ctype = mimetypes.guess_type(str(local))[0] or "application/octet-stream"
                if local.suffix == ".js":
                    ctype = "text/javascript"
                return resp(local.read_bytes(), ctype)
            return resp(b"not found", "text/plain", 404)
        except Exception as e:
            return resp(f"error: {e}".encode(), "text/plain", 500)

    # ---- WebSocket
    async def ws(self, connection):
        self.clients.add(connection)
        loop = asyncio.get_running_loop()
        try:
            await connection.send(json.dumps(self.sim.status()))
            async for message in connection:
                try:
                    cmd = json.loads(message)
                except Exception:
                    continue
                fut = loop.create_future()
                self.sim.commands.put((cmd, lambda out, f=fut: loop.call_soon_threadsafe(f.set_result, out)))
                out = await fut
                if out.get("type") in ("info", "type", "error"):
                    await connection.send(json.dumps(out))
                else:
                    status = self.sim.status()
                    status["ack"] = cmd.get("cmd")
                    if "seq" in cmd:
                        status["seq"] = cmd["seq"]
                    await connection.send(json.dumps(status))
                    status.pop("ack"); status.pop("seq", None)
                    await self.broadcast(json.dumps(status), exclude=connection)
        finally:
            self.clients.discard(connection)

    async def broadcast(self, payload, exclude=None) -> None:
        dead = []
        for c in list(self.clients):
            if c is exclude:
                continue
            try:
                await c.send(payload)
            except Exception:
                dead.append(c)
        for c in dead:
            self.clients.discard(c)

    async def pump(self) -> None:
        """Move frames from the simulation thread to the sockets."""
        last_status = 0.0
        while True:
            try:
                frame = self.sim.frames.get_nowait()
            except Empty:
                await asyncio.sleep(0.005)
                continue
            if self.clients:
                await self.broadcast(frame)
                now = time.time()
                if now - last_status > 0.5:
                    last_status = now
                    await self.broadcast(json.dumps(self.sim.status()))

    async def serve(self, host: str, port: int) -> None:
        from websockets.asyncio.server import serve

        self.sim.start()
        async with serve(self.ws, host, port, process_request=self.http, max_size=None, compression=None):
            print(f"NeuroFly gallery: http://{host}:{port}  ({self.brain.n} neurons)", file=sys.stderr, flush=True)
            await self.pump()


def main(host: str = "0.0.0.0", port: int = 8765, window_ms: float = 10.0, seed: int | None = None,
         data_dir=None, release=None, prefetch_meshes: bool = True) -> None:
    brain = load_brain(data_dir, seed=seed, release=release)
    atlas = Atlas(brain.ids, data_dir)
    if prefetch_meshes:
        from .archive import fetch_meshes
        try:
            fetch_meshes(["volume"])
        except Exception as e:
            print(f"brain mesh unavailable ({e}); the screen works without it", file=sys.stderr)
    server = GalleryServer(brain, atlas, window_ms)
    asyncio.run(server.serve(host, port))
