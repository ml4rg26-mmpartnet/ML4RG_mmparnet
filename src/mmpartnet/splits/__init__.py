"""mmpartnet.splits: swappable train/test split axis (CONTRACT.md swap-in #4).

    from mmpartnet.splits import SplitConfig, get_split, holdout_chrom
    sp = get_split(rbps, SplitConfig(axis="family", held_family="RRM"))
    sp.train, sp.test                                  # RBP symbol tuples
    tr_idx, te_idx = holdout_chrom(windows, "chr1")    # naive window-level holdout
"""
from __future__ import annotations
from .base import SplitConfig, RbpSplit, holdout_chrom
from .registry import get_split, list_splits, register

__all__ = ["SplitConfig", "RbpSplit", "holdout_chrom", "get_split", "list_splits", "register"]
