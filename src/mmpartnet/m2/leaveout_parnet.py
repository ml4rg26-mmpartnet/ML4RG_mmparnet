"""Swap-in #1: the leave-out-pretrained PARNET (the DECISIVE experiment), mostly contract.

`load_parnet()` is already weight-agnostic, so flipping weights is one env var: ML4RG_PARNET_WEIGHTS
-> a leave-out-pretraining checkpoint (a PARNET trained with the held RBPs EXCLUDED). Only on
those weights is rbp_holdout (splits #4) an honest zero-shot test; on the leaked all-223 weights the
body has already seen every RBP. This module is the seam + a provenance guard; the gated artifact is
the checkpoint itself (ask a supervisor).
"""
from __future__ import annotations
import os


def load_leaveout_parnet(held_rbps, weights=None, device=None):
    """Load a leave-out-pretrained PARNET and guard its provenance.

    Contract: returns a models.parnet.ParnetModel exactly like load_parnet (weight-agnostic), but
    REFUSES to silently pass off the leaked all-223 checkpoint as a leave-out model. Set
    ML4RG_PARNET_WEIGHTS (or `weights`) to the lab's leave-out checkpoint AND record which RBPs were
    excluded so callers can assert held_rbps subset of the excluded set.
    """
    from ..models.parnet import load_parnet
    w = str(weights or os.environ.get("ML4RG_PARNET_WEIGHTS", ""))
    excluded = os.environ.get("ML4RG_PARNET_HELDOUT", "")   # comma-list the lab pretrained WITHOUT
    if not excluded:
        raise NotImplementedError(
            "leave-out PARNET not provided. Set ML4RG_PARNET_WEIGHTS to the lab's leave-out checkpoint "
            "and ML4RG_PARNET_HELDOUT to the RBPs it EXCLUDED, so rbp_holdout zero-shot is honest. "
            "Until then use the all-223 weights for in-distribution work only (NOT a zero-shot claim).")
    exc = {g.strip().upper() for g in excluded.split(",") if g.strip()}
    missing = [g for g in held_rbps if g.upper() not in exc]
    if missing:
        raise ValueError(f"held RBPs not in the checkpoint's excluded set (leakage!): {missing}")
    return load_parnet(weights=w, device=device)
