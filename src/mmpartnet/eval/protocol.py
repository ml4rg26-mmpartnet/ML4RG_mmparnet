"""The evaluation protocol — model-agnostic leave-out-RBP gate + the family-disjoint guarantee.

This factors the reusable core out of ``experiments.binding_gate`` so a teammate evaluates ANY head the
same way: give a ``score_fn(feats, protein_vec) -> per-window scores`` and the held-out-RBP set, and get
back per-RBP AUROC/AUPRC, the CORAL fraction>bar, seen/unseen means, and the mandatory control deltas
(protein-shuffle / within-family) each B3 fire-checked. Nothing here trains — you pass a trained head's
scorer. ``torch_head_scorer`` adapts a torch head; ``family_disjoint_assert`` enforces zero family
overlap between the train and eval RBP sets (the split-level leakage guarantee).

Honesty gate: results carry ``honest_zero_shot`` (config.honest_zero_shot()) — False on the default
all-223 leaked PARNET body, so a reader never mistakes a proxy number for a real held-out claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import roc_auc, average_precision, partition_seen_unseen
from .controls import shuffle_indices, within_family_indices, control_fired


@dataclass
class EvalResult:
    real_mean: float
    frac_above_bar: float
    bar: float
    n_held: int
    seen_unseen: dict
    controls: dict
    honest_zero_shot: bool
    rows: list = field(default_factory=list)


def torch_head_scorer(head, sigmoid=True, logit_index=0):
    """Adapt a torch head to a ``score_fn(feats, protein_vec) -> np.ndarray``. Assumes the head returns
    a logit (or a tuple whose ``logit_index`` element is the logit) given (feats, protein_vec expanded to
    feats' batch). Wrap your own head if its call convention differs."""
    import torch

    def score_fn(feats, protein_vec):
        with torch.no_grad():
            pv = protein_vec.expand(len(feats), -1) if protein_vec.dim() == 1 else protein_vec
            out = head(feats, pv)
            logit = out[logit_index] if isinstance(out, (tuple, list)) else out
            logit = logit.squeeze(-1) if logit.dim() > 1 else logit
            return (torch.sigmoid(logit) if sigmoid else logit).detach().cpu().numpy()

    return score_fn


def _eval_held(score_fn, feats, held_k, emap, te_y, ti_keep, syms, fam_lab, min_pos=5):
    rows = []
    for k in held_k:
        y = (te_y[:, ti_keep[k]] > 0).astype(float)
        if y.sum() < min_pos:
            continue
        s = np.asarray(score_fn(feats, emap[k]))
        rows.append({"rbp": syms[k], "family": fam_lab[k],
                     "auroc": roc_auc(s, y), "auprc": average_precision(s, y),
                     "pos_rate": float(y.mean())})
    return rows


def held_rbp_gate(score_fn, feats, held_k, prot_emb, te_y, ti_keep, syms, fam_lab,
                  train_names=None, bar=0.65, seed=0, min_gap=0.05, honest_zero_shot=None):
    """Model-agnostic M1 leave-out-RBP gate.

    score_fn(feats, protein_vec)->scores; feats: (W, d) window features from the frozen backbone;
    held_k: track-indices of the HELD-OUT RBPs; prot_emb: indexable protein reps (k -> vector);
    te_y: (W, T) label matrix; ti_keep: k -> column in te_y; syms/fam_lab: per-k name/family.

    Runs REAL + protein-shuffle + within-family controls, each fire-checked (a shuffle that does not
    drop AUROC toward chance is flagged: leaky backbone or protein-ignoring head)."""
    rng = np.random.default_rng(seed)
    real = _eval_held(score_fn, feats, held_k, {k: prot_emb[k] for k in held_k},
                      te_y, ti_keep, syms, fam_lab)

    # protein-shuffle: derange the reps across held RBPs
    perm = shuffle_indices(len(held_k), seed=seed)
    shuf_map = {held_k[i]: prot_emb[held_k[perm[i]]] for i in range(len(held_k))}
    shuf = _eval_held(score_fn, feats, held_k, shuf_map, te_y, ti_keep, syms, fam_lab)

    # within-family shuffle (harder)
    fam_ids = np.array([hash(fam_lab[k]) % (10 ** 8) for k in held_k])
    wperm = within_family_indices(fam_ids, seed=seed + 1)
    wf_map = {held_k[i]: prot_emb[held_k[wperm[i]]] for i in range(len(held_k))}
    wfam = _eval_held(score_fn, feats, held_k, wf_map, te_y, ti_keep, syms, fam_lab)

    def _m(rows):
        a = np.array([r["auroc"] for r in rows], float)
        return float(np.nanmean(a)) if len(a) else float("nan")

    ar = np.array([r["auroc"] for r in real], float)
    frac = float(np.mean(ar > bar)) if len(ar) else float("nan")
    controls = {
        "protein_shuffle": {"mean_auroc": _m(shuf),
                            **control_fired(_m(real), _m(shuf), min_gap, "down")},
        "within_family": {"mean_auroc": _m(wfam),
                          **control_fired(_m(real), _m(wfam), min_gap, "down")},
    }
    su = partition_seen_unseen(real, train_names or [], value_key="auroc")
    if honest_zero_shot is None:
        try:
            from .. import config
            honest_zero_shot = config.honest_zero_shot()
        except Exception:
            honest_zero_shot = False
    return EvalResult(real_mean=_m(real), frac_above_bar=frac, bar=bar, n_held=len(real),
                      seen_unseen=su, controls=controls, honest_zero_shot=bool(honest_zero_shot),
                      rows=real)


def run_controls(real_rows, control_rows, value_key="auroc", min_gap=0.05, directions=None):
    """Given the REAL per-RBP rows and a dict {control_name: rows}, compute each control's mean + the
    B3 fire-check + the delta. directions maps control_name -> 'down'|'up' (default 'down')."""
    directions = directions or {}

    def _m(rows):
        a = np.array([r[value_key] for r in rows], float)
        return float(np.nanmean(a)) if len(a) else float("nan")

    real = _m(real_rows)
    out = {}
    for name, rows in control_rows.items():
        d = directions.get(name, "down")
        out[name] = {"mean": _m(rows), **control_fired(real, _m(rows), min_gap, d)}
    return {"real_mean": real, "controls": out}


def family_disjoint_assert(train_names, eval_names, family_of):
    """Enforce the split-level leakage guarantee: NO family may appear in both the train and the eval
    RBP sets. `family_of`: name -> family id. Raises AssertionError listing the offending families."""
    tf = {family_of[n] for n in train_names if n in family_of}
    ef = {family_of[n] for n in eval_names if n in family_of}
    overlap = tf & ef
    assert not overlap, (f"family leakage: {len(overlap)} families in BOTH train and eval "
                         f"({sorted(list(overlap))[:8]}{'...' if len(overlap) > 8 else ''}); "
                         f"use a family-disjoint split (splits/strategies/family.py).")
    return True
