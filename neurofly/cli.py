"""Command line entry point: ``neurofly download | info | run``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import neurons as N


def _parse_neurons(spec: list[str], release: str) -> list[int]:
    """'sugar' / 'p9' / 'mn9' names or raw FlyWire ids, mixed freely."""
    out: list[int] = []
    for s in spec:
        if s.lower() in N.NAMED:
            out.extend(N.named(s, release))
        else:
            out.append(int(s))
    return out


def cmd_download(args: argparse.Namespace) -> None:
    from .data import download, load_connectome

    d = download(args.data_dir, force=args.force, release=args.release)
    ids, W = load_connectome(d, release=args.release)
    print(f"ok: {len(ids)} neurons, {W.nnz} connections, cached in {d}")


def cmd_info(args: argparse.Namespace) -> None:
    from .data import load_connectome

    ids, W = load_connectome(args.data_dir, release=args.release)
    print(f"neurons:      {len(ids)}")
    print(f"connections:  {W.nnz} (neuron pairs)")
    print(f"synapses:     {int(abs(W.data).sum())}")
    print(f"excitatory:   {(W.data > 0).mean() * 100:.1f} % of connections")
    print(f"out-degree:   mean {W.nnz / len(ids):.1f}")
    for name, ids_ in N.NAMED.items():
        print(f"set '{name}':  {len(ids_)} neurons")


def cmd_run(args: argparse.Namespace) -> None:
    from .data import load_brain
    from .model import SpikeRecord

    t0 = time.perf_counter()
    brain = load_brain(args.data_dir, seed=args.seed, release=args.release)
    print(f"loaded {brain.n} neurons in {time.perf_counter() - t0:.1f} s", file=sys.stderr)

    rel = brain.release
    stim = _parse_neurons(args.activate, rel)
    brain.activate(stim, rate_hz=args.rate)
    if args.activate2:
        brain.activate(_parse_neurons(args.activate2, rel), rate_hz=args.rate2)
    if args.silence:
        brain.silence(_parse_neurons(args.silence, rel))
    watch = _parse_neurons(args.watch, rel) if args.watch else []

    print(
        f"activating {len(stim)} neurons at {args.rate:g} Hz, "
        f"{len(args.silence or [])} silenced, {args.trials} x {args.t:g} ms",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    records = brain.run_trials(args.t, args.trials, progress=args.progress)
    wall = time.perf_counter() - t0
    bio = args.t * args.trials / 1000
    print(f"simulated {bio:g} s in {wall:.1f} s wall ({wall / bio:.1f} x slower than real time)", file=sys.stderr)

    table = SpikeRecord.mean_rates(records)
    n_spikes = sum(len(r) for r in records) / len(records)
    print(f"\n{n_spikes:.0f} spikes per trial, {len(table)} active neurons\n")

    print("top neurons (Hz):")
    top = table.head(args.top)
    for fid, row in top.iterrows():
        print(f"  {fid}  {N.label(fid):>10}  {row.rate_hz:7.1f} +- {row.std_hz:5.1f}")

    if watch:
        print("\nwatched neurons (Hz):")
        for fid in watch:
            r = table.rate_hz.get(fid, 0.0)
            s = table.std_hz.get(fid, 0.0)
            print(f"  {fid}  {N.label(fid):>10}  {r:7.1f} +- {s:5.1f}")

    if args.out:
        import pandas as pd

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.concat([r.to_frame() for r in records], ignore_index=True)
        df["exp_name"] = out.stem
        df.to_parquet(out)
        print(f"\nspikes written to {out}", file=sys.stderr)


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import main as serve_main

    serve_main(host=args.host, port=args.port, window_ms=args.window, seed=args.seed,
               data_dir=args.data_dir, release=args.release, prefetch_meshes=not args.no_meshes)


def cmd_archive(args: argparse.Namespace) -> None:
    from .archive import fetch_meshes, fetch_skeletons
    from .data import load_connectome

    ids, _ = load_connectome(args.data_dir, release=args.release)
    if args.meshes:
        fetch_meshes()
    if args.skeletons:
        n = len(ids) if args.skeletons == "all" else int(args.skeletons)
        print(f"fetching {n} skeletons...", file=sys.stderr)
        fetch_skeletons(ids[:n], args.data_dir, workers=args.workers)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="neurofly", description=__doc__)
    ap.add_argument("--data-dir", default=None, help="where the connectome lives (default: $NEUROFLY_DATA or ./data)")
    ap.add_argument("--release", default=None, choices=["783", "630"], help="FlyWire release (default: $NEUROFLY_RELEASE or 783)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch the connectome and build the cache")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("info", help="print connectome statistics")
    i.set_defaults(func=cmd_info)

    r = sub.add_parser("run", help="activate / silence neurons and report firing rates")
    r.add_argument("--activate", nargs="+", default=["sugar"], help="set names (sugar, p9) or FlyWire ids")
    r.add_argument("--rate", type=float, default=200.0, help="Poisson rate in Hz (default 200)")
    r.add_argument("--activate2", nargs="+", default=None, help="second set with its own --rate2")
    r.add_argument("--rate2", type=float, default=100.0)
    r.add_argument("--silence", nargs="+", default=None, help="neurons whose output is muted")
    r.add_argument("--watch", nargs="+", default=["mn9"], help="neurons to report explicitly")
    r.add_argument("--t", type=float, default=1000.0, help="trial length in ms (default 1000)")
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--top", type=int, default=15)
    r.add_argument("--out", default=None, help="write all spikes to this parquet file")
    r.add_argument("--progress", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="start the gallery server (simulation + web viewer)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--window", type=float, default=10.0, help="ms of model time per streamed frame")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--no-meshes", action="store_true", help="do not download the brain outline mesh")
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("archive", help="prefetch the public FlyWire archive (meshes, skeletons) for offline use")
    a.add_argument("--meshes", action="store_true", help="brain outline and 75 neuropil meshes (~10 MB)")
    a.add_argument("--skeletons", default=None, help="number of skeletons to fetch, or 'all' (~139k files, several GB)")
    a.add_argument("--workers", type=int, default=8)
    a.set_defaults(func=cmd_archive)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
