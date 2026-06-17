"""POPULATED: naive axis. No RBP holdout (every RBP is in both train and test); the actual
held-out evaluation is at the WINDOW level by chromosome (see base.holdout_chrom). This measures
in-distribution profile fit, the demo's default."""
from __future__ import annotations
from ..base import RbpSplit
from ..registry import register


@register("naive")
def naive(rbps, cfg, meta=None) -> RbpSplit:
    t = tuple(rbps)
    return RbpSplit(train=t, test=t, axis="naive",
                    note=f"no RBP holdout; window-level holdout on {cfg.held_chrom}")
