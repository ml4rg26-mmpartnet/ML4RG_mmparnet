"""POPULATED: pooled ProtT5 embeddings stored in an H5 file.

This provider adapts the manually gathered ProtT5 reduced embedding H5 to the
repository's swappable ``mmpartnet.protein`` interface.  The FiLM experiments
primarily resolve embeddings by the exact H5 key recorded in
``mmpartnet_out/prott5_track_map.tsv``; ``vector(rbp)`` is also provided for the
standard per-symbol registry API.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from ..base import ProteinSource
from ..registry import register


DEFAULT_PROTT5_H5 = (
    Path("/home/dgu/storage_ml4rg26-mmparnet")
    / "manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5"
)


@register("prott5_h5")
class ProtT5H5(ProteinSource):
    """Pooled ProtT5 vectors addressed by RBP symbol or H5 key."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.h5_path = Path(self.cfg.extra.get("h5_path", DEFAULT_PROTT5_H5))
        self.track_map = self.cfg.extra.get("track_map")
        self._h5 = None
        self._symbol_to_key = None

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _load_symbol_to_key(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if self.track_map is None:
            return mapping
        with Path(self.track_map).open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("status") != "matched":
                    continue
                key = row.get("h5_key")
                if not key:
                    continue
                for symbol in (row.get("rbp"), row.get("match_gene")):
                    if symbol and symbol not in mapping:
                        mapping[symbol] = key
        return mapping

    @property
    def symbol_to_key(self) -> dict[str, str]:
        if self._symbol_to_key is None:
            self._symbol_to_key = self._load_symbol_to_key()
        return self._symbol_to_key

    def vector_by_key(self, key: str) -> Optional[np.ndarray]:
        if key not in self.h5:
            return None
        return np.asarray(self.h5[key][()], dtype=np.float32).ravel()

    def vector(self, rbp: str) -> Optional[np.ndarray]:
        key = self.symbol_to_key.get(rbp, rbp)
        return self.vector_by_key(str(key))
