"""Control / null registry + the B3 fire-check.

The reusable primitives (protein-shuffle nulls, random-body control, RNA-only baseline, paired stats)
live in ``experiments.eval_controls`` — the single source of truth — and are re-exported here so a
teammate imports ``mmpartnet.eval`` and gets the SAME nulls everyone else uses.

CONTROLS documents each null + the direction it must move the metric. ``control_fired`` enforces the
**B3 discipline**: a control that does NOT move the metric is a *broken evaluation* (the head is not
using the signal the control removes), not a pass — so the harness flags it loudly instead of silently
reporting a "beat".
"""
from __future__ import annotations

import numpy as np

# re-export the primitives (single source of truth = experiments.eval_controls)
from mmpartnet.experiments.eval_controls import (  # noqa: F401
    derangement, within_family_perm, RandomBody, train_rna_only_multi,
    boot_ci, sign_test, summarize_method, feats_pos, pooled, auprc_cols,
)


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
