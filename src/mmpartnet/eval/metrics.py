"""Metrics - dependency-light (no sklearn), shared by every head + a teammate's model.

roc_auc / average_precision are rank-based so results are reproducible without sklearn version drift.
profile_pearson is the M2 nt-resolution metric. partition_seen_unseen splits any per-RBP metric list
into the RBPs the head trained on (seen) vs the held-out RBPs (unseen) - the zero-shot number.

Self-contained: no dependency on any experiment module, so the eval contract is a clean plug-in surface.
"""
from __future__ import annotations

import numpy as np


def average_precision(scores, y) -> float:
    """Area under the precision-recall curve (sklearn-free). y binary."""
    scores = np.asarray(scores, float)
    y = np.asarray(y, float)
    if y.sum() < 1:
        return float("nan")
    order = np.argsort(-scores)
    yo = y[order]
    tp = np.cumsum(yo)
    fp = np.cumsum(1 - yo)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / y.sum()
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def roc_auc(score, y) -> float:
    """Rank-based AUROC (Mann-Whitney), tie-averaged. y in {0,1}. NaN if a class is empty."""
    y = np.asarray(y)
    s = np.asarray(score, float)
    n1 = int((y > 0).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg = {}
    start = 0
    for i, c in enumerate(cnt):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    return float((ranks[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def profile_pearson(pred, obs, mask=None) -> float:
    """Per-position Pearson between a predicted and observed profile (1-D), the M2 metric.
    `mask` (bool, same length) restricts to valid positions. NaN if degenerate."""
    pred = np.asarray(pred, float).ravel()
    obs = np.asarray(obs, float).ravel()
    if mask is not None:
        mask = np.asarray(mask, bool).ravel()
        pred, obs = pred[mask], obs[mask]
    if len(pred) < 3 or np.std(pred) < 1e-12 or np.std(obs) < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, obs)[0, 1])


def partition_seen_unseen(rows, train_names, name_key="rbp", value_key="auroc"):
    """Split per-RBP metric `rows` (list of dicts) into seen (trained) vs unseen (held-out) by name.
    Returns {"seen": mean, "unseen": mean, "n_seen": int, "n_unseen": int} - 'unseen' is the zero-shot
    number that matters; 'seen' is the in-distribution (RBP-identity-capable) number."""
    train = set(train_names)
    seen = [r[value_key] for r in rows if r.get(name_key) in train and r[value_key] == r[value_key]]
    unseen = [r[value_key] for r in rows if r.get(name_key) not in train and r[value_key] == r[value_key]]
    return {
        "seen": float(np.mean(seen)) if seen else float("nan"),
        "unseen": float(np.mean(unseen)) if unseen else float("nan"),
        "n_seen": len(seen),
        "n_unseen": len(unseen),
    }
