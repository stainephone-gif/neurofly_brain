"""Download and load the FlyWire (v783) connectome used by the model.

Two files are needed, both taken from the public mirror in the
`eonsystemspbc/fly-brain` repository (the same files ship with the
original Shiu et al. code, remapped to FlyWire release 783):

* ``2025_Completeness_783.csv``      list of neuron IDs (row order = index)
* ``2025_Connectivity_783.parquet``  one row per connected neuron pair with
  the synapse count and the sign (+1 excitatory / -1 inhibitory)

The first load builds a compressed sparse column matrix (columns are
presynaptic neurons) and caches it as ``connectome_<release>.npz`` next to
the raw files, so later loads take about a second.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .model import FlyBrain, Params

# FlyWire release -> (neuron list, connectivity) file names and public mirrors.
# 783 (default): the current public release, from the Eon Systems port.
# 630: the release used in the Shiu et al. paper, from the original repo;
#      handy to compare with the reference results shipped there.
RELEASES = {
    "783": {
        "neurons": "2025_Completeness_783.csv",
        "connectivity": "2025_Connectivity_783.parquet",
        "base": "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data/",
    },
    "630": {
        "neurons": "2023_03_23_completeness_630_final.csv",
        "connectivity": "2023_03_23_connectivity_630_final.parquet",
        "base": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/",
    },
}
DEFAULT_RELEASE = "783"


def _release(release: str | None) -> str:
    release = str(release or os.environ.get("NEUROFLY_RELEASE", DEFAULT_RELEASE))
    if release not in RELEASES:
        raise ValueError(f"unknown release {release!r}, choose from {list(RELEASES)}")
    return release


def default_data_dir() -> Path:
    """Directory for the raw data: ``$NEUROFLY_DATA`` or ``./data``."""
    return Path(os.environ.get("NEUROFLY_DATA", "data")).expanduser()


def _fetch(url: str, dest: Path, attempts: int = 4) -> None:
    """Download ``url`` to ``dest``; verifies the size and retries on short reads."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "neurofly/0.1"})
    for attempt in range(1, attempts + 1):
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    sys.stderr.write(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
            sys.stderr.write("\n")
        if not total or done == total:
            tmp.replace(dest)
            return
        print(f"  short read ({done} of {total} bytes), retry {attempt}/{attempts}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
    raise IOError(f"could not download {url} completely after {attempts} attempts")


def download(data_dir: Path | str | None = None, force: bool = False, release: str | None = None) -> Path:
    """Download the two raw files into ``data_dir`` (skips existing files)."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    files = RELEASES[_release(release)]
    for key in ("neurons", "connectivity"):
        dest = data_dir / files[key]
        if dest.exists() and not force:
            continue
        url = files["base"] + files[key]
        print(f"downloading {url}", file=sys.stderr)
        _fetch(url, dest)
    return data_dir


def _build_cache(data_dir: Path, release: str) -> Path:
    import pandas as pd

    files = RELEASES[release]
    neurons_csv = data_dir / files["neurons"]
    conn_parquet = data_dir / files["connectivity"]
    print("building sparse connectome (one-off, ~10 s)...", file=sys.stderr)

    ids = pd.read_csv(neurons_csv, index_col=0).index.to_numpy(dtype=np.int64)
    n = len(ids)

    con = pd.read_parquet(
        conn_parquet,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = con["Presynaptic_Index"].to_numpy()
    post = con["Postsynaptic_Index"].to_numpy()
    w = con["Excitatory x Connectivity"].to_numpy(dtype=np.float32)

    # rows = postsynaptic, columns = presynaptic  ->  g += W @ spikes
    W = sp.csc_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()

    cache = data_dir / f"connectome_{release}.npz"
    np.savez(cache, ids=ids, indptr=W.indptr, indices=W.indices, data=W.data)
    return cache


def load_connectome(data_dir: Path | str | None = None, release: str | None = None) -> tuple[np.ndarray, sp.csc_matrix]:
    """Return ``(flywire_ids, W)``.

    ``W`` is an ``n x n`` CSC matrix of *signed synapse counts*
    (``W[post, pre]``). Multiply by ``Params.w_syn`` to get millivolts.
    Downloads the raw data and builds the cache when missing.
    ``release`` is "783" (default, or ``$NEUROFLY_RELEASE``) or "630".
    """
    release = _release(release)
    data_dir = download(data_dir, release=release)
    cache = data_dir / f"connectome_{release}.npz"
    if not cache.exists():
        _build_cache(data_dir, release)
    z = np.load(cache)
    n = len(z["ids"])
    W = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    return z["ids"], W


def load_brain(
    data_dir: Path | str | None = None,
    params: Params | None = None,
    seed: int | None = None,
    release: str | None = None,
) -> FlyBrain:
    """Build a ready-to-run :class:`FlyBrain` from the cached connectome."""
    release = _release(release)
    ids, W = load_connectome(data_dir, release=release)
    return FlyBrain(W, ids, params=params, seed=seed, release=release)
