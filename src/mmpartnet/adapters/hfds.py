"""Swap-in #3 — the lab's canonical 600-nt eCLIP profile substrate.

`adapters/peaks.py` (built) turns PUBLIC ENCODE eCLIP BED peaks into ParnetDataElement
objects (the dev/now substrate). This module mirrors its API so that when the lab
delivers the canonical `encode.filtered.hfds` (HuggingFace datasets, the 223-track
600-nt profiles PARNET was trained on), switching is a one-line config change
(RunConfig.substrate = "peaks" -> "hfds") with ZERO downstream edits — every consumer
(train/eval/interpret) already speaks ParnetDataElement.

STATUS: stub. The HFDS format is lab-private and not yet delivered (gated). Keep the
function signatures byte-identical to `adapters.peaks` so the swap stays mechanical:

  rbp_elements(rbp, ...) -> list[ParnetDataElement]
  corpus(rbps, ...)      -> {rbp: [ParnetDataElement]}
  n2_windows(rbps, ...)  -> N2 (other-RBP-peak) negatives

CONFIRM with a supervisor before implementing: HFDS schema (feature keys for sequence /
eCLIP target / control), the negatives convention, and the cell-line track index.
"""
from __future__ import annotations

from mmpartnet.contract import ParnetDataElement  # noqa: F401  (the shared contract)

_NOT_IMPLEMENTED = (
    "adapters.hfds is a gated swap-in stub — the canonical encode.filtered.hfds is "
    "lab-private and not yet delivered. Use RunConfig.substrate='peaks' (public ENCODE "
    "eCLIP via adapters.peaks) until the lab's HFDS lands; then implement this module to "
    "the same signatures and flip the one config flag."
)


def rbp_elements(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED)


def corpus(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED)


def n2_windows(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED)
