"""Leakage-controlled, family-disjoint evaluation — the reusable core.

This is the project's differentiator packaged so a teammate imports it instead of copying it:
"beating the baseline" must mean beating a *leakage-controlled* baseline, on a *family-disjoint* split.

    from mmpartnet.eval import held_rbp_gate, run_controls, family_disjoint_assert
    from mmpartnet.eval import roc_auc, profile_pearson, partition_seen_unseen
    from mmpartnet.eval import CONTROLS, shuffle_indices, control_fired

Layers:
  metrics.py    rank AUROC, average precision, per-nt profile Pearson, seen/unseen partition
  controls.py   the null/baseline registry (protein-shuffle, within-family, RNA-only, random-body) + the
                B3 fire-check (a control that does NOT move the metric is a BROKEN eval, and is flagged)
  decompose.py  in-distribution (protein = RBP-identity lookup) vs zero-shot decomposition
  protocol.py   held_rbp_gate(score_fn, ...) — the M1 leave-out-RBP gate as a model-agnostic function;
                family_disjoint_assert(...) — the zero-family-overlap guarantee
"""
from __future__ import annotations

from .metrics import roc_auc, average_precision, profile_pearson, partition_seen_unseen
from .controls import (CONTROLS, shuffle_indices, within_family_indices, control_fired,
                       derangement, within_family_perm, RandomBody, boot_ci, sign_test, summarize_method)
from .protocol import EvalResult, held_rbp_gate, run_controls, family_disjoint_assert

__all__ = [
    "roc_auc", "average_precision", "profile_pearson", "partition_seen_unseen",
    "CONTROLS", "shuffle_indices", "within_family_indices", "control_fired",
    "derangement", "within_family_perm", "RandomBody", "boot_ci", "sign_test", "summarize_method",
    "EvalResult", "held_rbp_gate", "run_controls", "family_disjoint_assert",
]
