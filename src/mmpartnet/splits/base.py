"""Split layer (CONTRACT.md swap-in #4), same shape as `mmpartnet.data`.

A split STRATEGY partitions RBPs into train/test along the chosen axis. The axis is the experiment's
generalization claim: `naive` (no RBP holdout; windows split by chromosome) measures in-distribution
profile fit; `family` (leave-one-family-out) measures cross-family transfer; `rbp_holdout` (clean
zero-shot, the decisive M2 axis) measures held-out-RBP generalization and pairs with the leave-out
PARNET weights (swap-in #1). Swap axis by name with zero downstream edits.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SplitConfig:
    axis: str = "naive"            # registry name: naive | family | rbp_holdout
    held_family: str = ""          # family axis: the held-out family (empty -> leave-out the first)
    held_rbps: tuple = ()          # rbp_holdout axis: the held-out RBP symbols
    held_chrom: str = "chr1"       # naive axis: chromosome held out at the window level
    test_frac: float = 0.3
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RbpSplit:
    """RBP-level partition. `train`/`test` are tuples of RBP symbols (may overlap only for `naive`)."""
    train: tuple
    test: tuple
    axis: str = "naive"
    note: str = ""


def holdout_chrom(windows, chrom: str):
    """Window-level naive split helper: indices of windows NOT on `chrom` (train) vs ON it (test).
    `windows` is any sequence of objects with a `.chrom` attribute (e.g. data.base.Window)."""
    tr, te = [], []
    for i, w in enumerate(windows):
        (te if getattr(w, "chrom", None) == chrom else tr).append(i)
    return tr, te
