"""CORAL-comparable metrics: F1 / AUROC with a SEEN vs UNSEEN (cold-start) partition, plus the
affinity ``validate_grid`` certificate. Dependency-light (numpy only) so a teammate can score their
predictions the same way we scored CORAL.

Context (verified this project): CORAL's decisive number is the *component-wise* split (0% protein/RNA
overlap = true cold-start). The released single checkpoint LEAKS across folds (F1 ~0.92 vs the paper's
0.65); a clean per-fold retrain reproduces ~0.57. So always report the UNSEEN block, never 'overall'.
"""
from __future__ import annotations

import numpy as np

from mmpartnet.eval.metrics import roc_auc


def _f1_at(pred, y, thr):
    yhat = (np.asarray(pred, float) >= thr).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def coral_f1_auroc(pred, y, seen_mask=None, thr=0.5, best_thr=False):
    """F1 + AUROC overall and (if seen_mask given) split into seen/unseen. seen_mask[i]=True means pair
    i shares a protein OR RNA with training (NOT cold-start); the UNSEEN block is the real claim."""
    pred = np.asarray(pred, float)
    y = np.asarray(y).astype(int)

    def block(idx):
        if idx.sum() == 0:
            return {"f1": float("nan"), "auroc": float("nan"), "n": 0, "pos": 0}
        p, yy = pred[idx], y[idx]
        t = thr
        if best_thr and len(np.unique(p)):
            t = float(max(np.unique(p), key=lambda z: _f1_at(p, yy, z)))
        return {"f1": _f1_at(p, yy, t), "auroc": roc_auc(p, yy),
                "n": int(idx.sum()), "pos": int(yy.sum()), "thr": float(t)}

    out = {"overall": block(np.ones(len(y), bool))}
    if seen_mask is not None:
        sm = np.asarray(seen_mask, bool)
        out["seen"] = block(sm)
        out["unseen"] = block(~sm)
    return out


def validate_grid(scores, dist, families=None, n_perm=1000, seed=0, near_q=0.33, far_q=0.67):
    """Affinity 'far>near' certificate. If binding affinity is real, near pairs (small `dist`) should
    score HIGHER than far pairs. A family-block permutation (shuffle scores only WITHIN a family) gives
    a null that respects family structure, so the p-value is not inflated by family-mean effects.
    Returns {effect (near-mean minus far-mean), p, n_near, n_far}."""
    scores = np.asarray(scores, float)
    dist = np.asarray(dist, float)
    lo, hi = np.quantile(dist, near_q), np.quantile(dist, far_q)
    near, far = scores[dist <= lo], scores[dist >= hi]
    if len(near) == 0 or len(far) == 0:
        return {"effect": float("nan"), "p": float("nan"), "n_near": int(len(near)), "n_far": int(len(far))}
    eff = float(np.nanmean(near) - np.nanmean(far))
    rng = np.random.default_rng(seed)
    fam = np.asarray(families) if families is not None else None
    null = np.empty(n_perm)
    for b in range(n_perm):
        s = scores.copy()
        if fam is None:
            s = rng.permutation(s)
        else:
            for fv in np.unique(fam):
                m = np.where(fam == fv)[0]
                s[m] = scores[rng.permutation(m)]
        null[b] = np.nanmean(s[dist <= lo]) - np.nanmean(s[dist >= hi])
    p = float((np.sum(null >= eff) + 1) / (n_perm + 1))
    return {"effect": eff, "p": p, "n_near": int(len(near)), "n_far": int(len(far)), "n_perm": n_perm}
