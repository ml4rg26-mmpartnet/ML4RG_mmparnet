"""STUB: lab canonical substrate (encode.filtered / a HuggingFace dataset of pre-tiled windows).

This is the gated 'data substrate' swap-in (config.RunConfig.substrate = "hfds"): the exact tiles +
targets the lab trains PARNET on, rather than our peak-centered public reconstruction. When the lab shares
it, fill the four methods to stream rows from `datasets.load_dataset(...)` (or a local arrow/parquet dir),
mapping each row's window -> sequence + observed per-nt target. Set cfg.target="hfds" and fill the matching
preprocess.register("hfds") in preprocess.py to match the dataset's stored target format.

cfg.extra is the place for {"path": ..., "split": ..., "name": ...}. See docs/DATA_INVENTORY.md row
"encode.filtered (HFDS)".
"""
from __future__ import annotations
from typing import Iterable, Optional

from ..base import DataSource, Window
from ..registry import register


@register("hfds")
class HfdsSource(DataSource):
    target_kind = "hfds"

    def rbps(self) -> list:
        raise NotImplementedError(
            "hfds: lab encode.filtered / HFDS substrate not available yet (lab-gated). Fill from "
            "datasets.load_dataset(cfg.extra['path']); see docs/DATA_INVENTORY.md.")

    def windows(self, rbp: str) -> Iterable[Window]:
        raise NotImplementedError("hfds.windows: yield one Window per stored tile for `rbp`.")

    def sequence(self, w: Window) -> Optional[str]:
        raise NotImplementedError("hfds.sequence: return the tile's stored sequence (or hg38 lookup).")

    def observed(self, rbp: str, w: Window):
        raise NotImplementedError("hfds.observed: return the tile's stored per-nt target (raw).")
