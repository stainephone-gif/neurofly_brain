"""The public FlyWire 783 archive: skeletons, meshes, synapses.

Everything lives in an open Google Cloud bucket maintained by the Lee lab
(no login needed)::

    gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/fafb_783/

* ``fafb_fafb_space_swc/<root_id>.swc``  a skeleton for every neuron (nm)
* ``obj/fafb14_volume_raw.obj``          the brain outline mesh
* ``obj/neuropils/fafb14_neuropil_<NAME>_raw.obj``  75 neuropil meshes
* ``fafb_783_synapses.parquet``          every synapse with coordinates (2 GB)
* ``fafb_783_simple_edgelist.feather``   neuron-to-neuron edge list

This module downloads what a viewer needs on demand and caches it under
``data/archive``. Skeletons are parsed into line segments; meshes are
parsed into vertex/face arrays. All coordinates are converted to um.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import _fetch, default_data_dir

BUCKET = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/fafb_783"

NEUROPILS = [
    "AL_L", "AL_R", "AME_L", "AME_R", "AMMC_L", "AMMC_R", "AOTU_L", "AOTU_R", "ATL_L", "ATL_R",
    "AVLP_L", "AVLP_R", "BU_L", "BU_R", "CAN_L", "CAN_R", "CRE_L", "CRE_R", "EB", "EPA_L", "EPA_R",
    "FB", "FLA_L", "FLA_R", "GA_L", "GA_R", "GNG", "GOR_L", "GOR_R", "IB_L", "IB_R", "ICL_L", "ICL_R",
    "IPS_L", "IPS_R", "LAL_L", "LAL_R", "LH_L", "LH_R", "LOP_L", "LOP_R", "LO_L", "LO_R", "MB_CA_L",
    "MB_CA_R", "MB_ML_L", "MB_ML_R", "MB_PED_L", "MB_PED_R", "MB_VL_L", "MB_VL_R", "ME_L", "ME_R",
    "NO", "PB", "PLP_L", "PLP_R", "PRW", "PVLP_L", "PVLP_R", "SAD", "SCL_L", "SCL_R", "SIP_L",
    "SIP_R", "SLP_L", "SLP_R", "SMP_L", "SMP_R", "SPS_L", "SPS_R", "VES_L", "VES_R", "WED_L", "WED_R",
]


def archive_dir(data_dir: Path | str | None = None) -> Path:
    d = (Path(data_dir) if data_dir else default_data_dir()) / "archive"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------- skeletons
def skeleton_path(root_id: int, data_dir: Path | str | None = None) -> Path:
    d = archive_dir(data_dir) / "swc"
    d.mkdir(exist_ok=True)
    dest = d / f"{int(root_id)}.swc"
    if not dest.exists():
        _fetch(f"{BUCKET}/fafb_fafb_space_swc/{int(root_id)}.swc", dest)
    return dest


def fetch_skeletons(root_ids: Iterable[int], data_dir: Path | str | None = None, workers: int = 8) -> list[Path]:
    """Download many skeletons in parallel (skips cached ones)."""
    ids = [int(i) for i in root_ids]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda i: skeleton_path(i, data_dir), ids))


def load_skeleton(root_id: int, data_dir: Path | str | None = None) -> dict:
    """Parse an SWC file into ``{"nodes": (n,3) um, "radius": (n,), "edges": (m,2)}``."""
    path = skeleton_path(root_id, data_dir)
    rows = np.loadtxt(path, comments="#", ndmin=2)
    node_id = rows[:, 0].astype(np.int64)
    xyz = (rows[:, 2:5] / 1000.0).astype(np.float32)
    radius = (rows[:, 5] / 1000.0).astype(np.float32)
    parent = rows[:, 6].astype(np.int64)
    lookup = {n: k for k, n in enumerate(node_id)}
    has_parent = parent >= 0
    child = np.flatnonzero(has_parent)
    par = np.array([lookup[p] for p in parent[has_parent]], dtype=np.int64)
    edges = np.stack([par, child], axis=1)
    return {"root_id": int(root_id), "nodes": xyz, "radius": radius, "edges": edges}


def skeleton_segments(root_id: int, data_dir: Path | str | None = None) -> np.ndarray:
    """``(m, 2, 3)`` float32 line segments in um, ready for a line renderer."""
    s = load_skeleton(root_id, data_dir)
    return s["nodes"][s["edges"]]


# --------------------------------------------------------------------- meshes
def mesh_path(name: str, data_dir: Path | str | None = None) -> Path:
    d = archive_dir(data_dir) / "obj"
    d.mkdir(exist_ok=True)
    if name == "volume":
        rel, fname = "obj/fafb14_volume_raw.obj", "fafb14_volume_raw.obj"
    else:
        if name not in NEUROPILS:
            raise KeyError(f"unknown neuropil {name!r}")
        fname = f"fafb14_neuropil_{name}_raw.obj"
        rel = f"obj/neuropils/{fname}"
    dest = d / fname
    if not dest.exists():
        _fetch(f"{BUCKET}/{rel}", dest)
    return dest


def load_mesh(name: str, data_dir: Path | str | None = None) -> dict:
    """Parse an OBJ into ``{"vertices": (n,3) um float32, "faces": (m,3) int32}``."""
    verts, faces = [], []
    with open(mesh_path(name, data_dir)) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return {
        "name": name,
        "vertices": (np.asarray(verts, dtype=np.float32) / 1000.0),
        "faces": np.asarray(faces, dtype=np.int32),
    }


def fetch_meshes(names: Iterable[str] | None = None, data_dir: Path | str | None = None, workers: int = 8) -> None:
    names = list(names) if names is not None else ["volume", *NEUROPILS]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda n: mesh_path(n, data_dir), names))
    print(f"{len(names)} meshes cached", file=sys.stderr)
