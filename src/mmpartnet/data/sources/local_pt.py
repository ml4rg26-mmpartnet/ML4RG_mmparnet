"""STUB: pre-tiled local shards (offline / CI / teammate handoff).

For running with no network: a directory of torch .pt shards, each a list of ParnetDataElement-like dicts
(contract.make_element output) or {window, sequence, observed}. Useful to freeze a fixed demo slice so the
notebook runs deterministically in CI, and for handing a teammate an exact reproducible window set.

To populate: glob cfg.extra["dir"]/*.pt, torch.load each, and map elements -> Window + sequence + observed.
Group/RBP filtering via io.groups as in encode_bigwig.
"""
from __future__ import annotations
from typing import Iterable, Optional

from ..base import DataSource, Window
from ..registry import register


@register("local_pt")
class LocalPtSource(DataSource):
    target_kind = "counts"

    def rbps(self) -> list:
        raise NotImplementedError(
            "local_pt: pre-tiled .pt shard source not filled yet. Glob cfg.extra['dir']/*.pt and "
            "index elements by rbp; see docs/DATA_INVENTORY.md.")

    def windows(self, rbp: str) -> Iterable[Window]:
        raise NotImplementedError("local_pt.windows: yield a Window per stored element for `rbp`.")

    def sequence(self, w: Window) -> Optional[str]:
        raise NotImplementedError("local_pt.sequence: return the element's stored sequence.")

    def observed(self, rbp: str, w: Window):
        raise NotImplementedError("local_pt.observed: return the element's stored per-nt target.")
