"""Leakage-controlled, family-disjoint evaluation contract.

A teammate evaluates ANY registered head the same way: build the head (models/registry.py), get a
``score_fn`` (or use ``torch_head_scorer``), and run ``held_rbp_gate`` / ``run_controls`` - which apply
the mandatory nulls (protein-shuffle, within-family), each B3 fire-checked, and stamp every result with
``honest_zero_shot`` so a proxy number is never mistaken for a real held-out claim.
"""
from __future__ import annotations

from .metrics import roc_auc, average_precision, profile_pearson, partition_seen_unseen
from .controls import (CONTROLS, control_fired, shuffle_indices, within_family_indices,
                       derangement, within_family_perm)
from .protocol import (EvalResult, torch_head_scorer, held_rbp_gate, run_controls,
                       family_disjoint_assert)

__all__ = [
    "roc_auc", "average_precision", "profile_pearson", "partition_seen_unseen",
    "CONTROLS", "control_fired", "shuffle_indices", "within_family_indices",
    "derangement", "within_family_perm",
    "EvalResult", "torch_head_scorer", "held_rbp_gate", "run_controls", "family_disjoint_assert",
]
