"""rbp_holdout axis (clean zero-shot, the DECISIVE M2 split). Hold out cfg.held_rbps entirely:
they appear in NO training window, so a positive test number is genuine held-RBP generalization.
Pairs with the leave-out PARNET weights (swap-in #1): only then is the zero-shot claim clean
(otherwise the all-223 PARNET body has already seen the held RBP). Functional given held_rbps;
the LAB-GATED part is obtaining a leave-out-pretrained PARNET so the test is honest (see m2/).
"""
from __future__ import annotations
from ..base import RbpSplit
from ..registry import register


@register("rbp_holdout")
def rbp_holdout(rbps, cfg, meta=None) -> RbpSplit:
    held = {g.upper() for g in cfg.held_rbps}
    if not held:
        raise ValueError(
            "rbp_holdout needs cfg.held_rbps (the zero-shot held-out RBP symbols), e.g. the IGF2BP1/2/3 "
            "paralog group as TEST and a non-motif group as CONTROL.")
    test = tuple(g for g in rbps if g.upper() in held)
    train = tuple(g for g in rbps if g.upper() not in held)
    note = "zero-shot; HONEST only with leave-out-pretrained PARNET (swap-in #1, lab-gated)"
    return RbpSplit(train=train, test=test, axis="rbp_holdout", note=note)
