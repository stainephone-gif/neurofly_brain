"""Names, classes, positions and transmitters for every neuron in the model.

Source: the FlyWire annotation table of Schlegel et al., Nature 2024
(``flyconnectome/flywire_annotations`` on GitHub, CC-BY 4.0), which covers
138 625 of the 138 639 neurons in release 783. It gives for each neuron a
3-D anchor point (and the soma when there is one), the hierarchy
``flow > super_class > cell_class > cell_sub_class > cell_type``, the
hemisphere and the predicted neurotransmitter.

Positions are returned in micrometres in the FAFB electron-microscopy
space (raw voxels are 4 x 4 x 40 nm), the same frame as the skeletons and
neuropil meshes in :mod:`neurofly.archive`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _fetch, default_data_dir

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
RAW_NAME = "flywire_neuron_annotations.tsv"
CACHE_NAME = "annotations_783.parquet"

VOXEL_NM = np.array([4.0, 4.0, 40.0])

COLUMNS = [
    "root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z",
    "flow", "super_class", "cell_class", "cell_sub_class", "cell_type",
    "hemibrain_type", "ito_lee_hemilineage", "top_nt", "top_nt_conf", "side", "nerve",
]


def download_annotations(data_dir: Path | str | None = None, force: bool = False) -> Path:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / RAW_NAME
    if not dest.exists() or force:
        print(f"downloading {ANNOTATIONS_URL}", file=sys.stderr)
        _fetch(ANNOTATIONS_URL, dest)
    return dest


def load_annotations(ids: np.ndarray, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Annotation table aligned with the model: one row per index in ``ids``.

    Columns: ``root_id, x, y, z`` (um, anchor point), ``soma_x/y/z`` (um,
    NaN when unknown), ``flow, super_class, cell_class, cell_sub_class,
    cell_type, hemibrain_type, hemilineage, nt, nt_conf, side, nerve`` and
    ``annotated`` (False for the handful of neurons the table lacks; their
    position is then the centre of the brain).
    """
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    cache = data_dir / CACHE_NAME
    if cache.exists():
        df = pd.read_parquet(cache)
        if len(df) == len(ids) and np.array_equal(df["root_id"].to_numpy(), ids):
            return df

    raw = download_annotations(data_dir)
    print("building annotation table...", file=sys.stderr)
    a = pd.read_csv(raw, sep="\t", usecols=COLUMNS, low_memory=False)
    a = a.drop_duplicates("root_id").set_index("root_id")
    a = a.reindex(ids)

    df = pd.DataFrame({"root_id": ids})
    df["annotated"] = a["pos_x"].notna().to_numpy()
    for axis, k in zip("xyz", range(3)):
        df[axis] = (a[f"pos_{axis}"].to_numpy() * VOXEL_NM[k] / 1000.0).astype(np.float32)
        df[f"soma_{axis}"] = (a[f"soma_{axis}"].to_numpy() * VOXEL_NM[k] / 1000.0).astype(np.float32)
    centre = df[["x", "y", "z"]].mean()
    for axis in "xyz":
        df[axis] = df[axis].fillna(centre[axis])
    for col in ["flow", "super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "side", "nerve"]:
        df[col] = a[col].fillna("").astype(str).to_numpy()
    df["hemilineage"] = a["ito_lee_hemilineage"].fillna("").astype(str).to_numpy()
    df["nt"] = a["top_nt"].fillna("").astype(str).to_numpy()
    df["nt_conf"] = a["top_nt_conf"].fillna(0.0).astype(np.float32).to_numpy()
    for col in ["flow", "super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "side", "nerve", "hemilineage", "nt"]:
        df[col] = df[col].astype("category")
    df.to_parquet(cache)
    return df


class Atlas:
    """Lookup helpers over the annotation table.

    >>> atlas = Atlas(brain.ids)
    >>> atlas.by_type("MDN")                      # 4 moonwalker descending neurons
    >>> atlas.by_type("DNa02", side="left")
    >>> atlas.search("Gr64")                      # cell types containing a string
    >>> atlas.describe(720575940660219265)        # 'CB0701 · motor · right · acetylcholine'
    """

    def __init__(self, ids: np.ndarray, data_dir: Path | str | None = None):
        self.df = load_annotations(np.asarray(ids, dtype=np.int64), data_dir)
        self.ids = self.df["root_id"].to_numpy()

    def _mask(self, column: str, value: str, side: str | None = None) -> np.ndarray:
        m = (self.df[column].astype(str) == value).to_numpy().copy()
        if side:
            m &= (self.df["side"].astype(str) == side).to_numpy()
        return m

    def by_type(self, cell_type: str, side: str | None = None) -> np.ndarray:
        """Indices of all neurons of a cell type (``hemibrain_type`` as fallback)."""
        m = self._mask("cell_type", cell_type, side)
        if not m.any():
            m = self._mask("hemibrain_type", cell_type, side)
        return np.flatnonzero(m)

    def by_class(self, value: str, level: str = "super_class", side: str | None = None) -> np.ndarray:
        """Indices by ``super_class`` / ``cell_class`` / ``cell_sub_class`` / ``flow``."""
        return np.flatnonzero(self._mask(level, value, side))

    def search(self, text: str, limit: int = 50) -> pd.DataFrame:
        """Cell types whose name contains ``text`` (case-insensitive) with counts."""
        ct = self.df["cell_type"].astype(str)
        hit = ct.str.contains(text, case=False, regex=False)
        return ct[hit].value_counts().head(limit).rename_axis("cell_type").reset_index(name="n")

    def describe(self, neuron: int) -> str:
        row = self.df.iloc[self.index(neuron)]
        parts = [str(row.cell_type) or str(row.hemibrain_type) or "?", str(row.super_class), str(row.side), str(row.nt)]
        return " · ".join(p for p in parts if p)

    def index(self, neuron: int) -> int:
        if neuron >= (1 << 40):
            hit = np.flatnonzero(self.ids == neuron)
            if not hit.size:
                raise KeyError(neuron)
            return int(hit[0])
        return int(neuron)

    def positions(self) -> np.ndarray:
        """``(n, 3)`` float32 array of anchor points in um."""
        return self.df[["x", "y", "z"]].to_numpy(dtype=np.float32)
