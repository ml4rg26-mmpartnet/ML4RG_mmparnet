"""Control / null registry + the B3 fire-check.

CONTROLS documents each null + the direction it must move the metric. ``control_fired`` enforces the
**B3 discipline**: a control that does NOT move the metric is a *broken evaluation* (the head is not
using the signal the control removes), not a pass - so the harness flags it loudly instead of silently
reporting a "beat".

Self-contained: the permutation primitives are defined here (no experiment-module dependency), so any
teammate importing ``mmpartnet.eval`` gets the SAME nulls everyone else uses.
"""
from __future__ import annotations

import numpy as np


# ── permutation primitives ───────────────────────────────────────────────────────────────────────
def derangement(K, rng):
    """A fixed-point-free permutation of range(K): no index keeps its original position."""
    while True:
        p = rng.permutation(K)
        if K < 2 or not np.any(p == np.arange(K)):
            return p


def within_family_perm(fam_ids, rng):
    """Permute only WITHIN each family group (indices sharing a fam_id), leaving cross-family fixed."""
    fam_ids = np.asarray(fam_ids)
    K = len(fam_ids)
    p = np.arange(K)
    for f in set(fam_ids.tolist()):
        idx = np.where(fam_ids == f)[0]
        if len(idx) >= 2:
            for _ in range(20):
                q = rng.permutation(idx)
                if not np.any(q == idx):
                    p[idx] = q
                    break
    return p


# ── control metadata (name -> what it removes + the direction it must move the metric) ───────────────
CONTROLS = {
    "protein_shuffle": {
        "removes": "the protein-identity signal (derangement of protein reps across RBPs)",
        "expect": "metric DROPS toward chance (real > shuffle); if not -> the head ignores the protein "
                  "OR the backbone leaks the RBP identity (broken zero-shot).",
        "direction": "down",
    },
    "within_family_perm": {
        "removes": "within-family protein-identity (permute reps only within an RBP family)",
        "expect": "harder control; metric drops if the head uses fine protein detail beyond family.",
        "direction": "down",
    },
    "rna_only": {
        "removes": "the protein entirely (track-aware RNA-only head, equal budget)",
        "expect": "conditioned method must BEAT this fair baseline (real > rna_only).",
        "direction": "up",
    },
    "random_body": {
        "removes": "the PARNET pretraining (frozen random-init body of the same shape)",
        "expect": "gap(rna_only - random_body) = the PARNET-leakage-attributable share of the baseline.",
        "direction": "up",
    },
    "family_mean_floor": {
        "removes": "everything but the family label (predict the family-mean profile/rate)",
        "expect": "conditioned method must beat the paralog/family floor to claim protein specificity.",
        "direction": "up",
    },
}


def shuffle_indices(n, seed=0):
    """Protein-shuffle permutation: a fixed-point-free derangement of range(n) (no RBP keeps its rep)."""
    return derangement(n, np.random.default_rng(seed))


def within_family_indices(fam_ids, seed=1):
    """Within-family permutation: shuffle protein reps only among RBPs sharing a family label."""
    return within_family_perm(np.asarray(fam_ids), np.random.default_rng(seed))


def control_fired(real_mean, control_mean, min_gap=0.05, direction="down") -> dict:
    """B3 fire-check. For a leakage null (direction='down') the control must pull the metric DOWN by at
    least `min_gap` (real - control >= min_gap). For a baseline the method must beat (direction='up')
    the method must exceed it by `min_gap`. Returns {fired, gap, warn}. warn=True => investigate: the
    control did not do its job (silent control == broken eval, not a clean win)."""
    real_mean = float(real_mean)
    control_mean = float(control_mean)
    gap = (real_mean - control_mean)
    fired = gap >= min_gap
    return {"fired": bool(fired), "gap": float(gap), "min_gap": float(min_gap),
            "direction": direction, "warn": (not fired),
            "note": ("OK: control moved the metric" if fired else
                     "WARN: control did NOT move the metric >= min_gap -> eval may be leaky/degenerate")}
