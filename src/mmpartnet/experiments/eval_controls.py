"""Reusable evaluation controls + baselines + statistics for protein-conditioned RBP binding — the
leakage-controlled-eval differentiator, factored into one importable module so every experiment (and a
teammate's model) shares the SAME nulls and stats. This is the core of the "beating the baseline must mean
beating a LEAKAGE-CONTROLLED baseline" contribution.

Controls/baselines:
  derangement / within_family_perm   protein-shuffle nulls (fixed-point-free; family-aware)
  RandomBody + train_rna_only_multi  the RNA-only track-aware baseline (real body) and the random-body
                                     control (quantifies the PARNET-leakage share of the baseline)
Stats:
  auprc_cols   per-RBP average precision
  sign_test    binomial + Wilcoxon on per-RBP paired deltas (honest direction)
  boot_ci      paired bootstrap with a live RNG
  feats_pos    frozen PARNET per-position features (downsampled to NPOS)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpartnet.process.onehot import batch_onehot
from mmpartnet.experiments.binding_eval import average_precision

NPOS = 64
EPOCHS = 12
LR = 5e-4


# ── protein-shuffle nulls ──────────────────────────────────────────────────────
def derangement(K, rng):
    while True:
        p = rng.permutation(K)
        if not np.any(p == np.arange(K)):
            return p


def within_family_perm(fam_ids, rng):
    K = len(fam_ids); p = np.arange(K)
    for f in set(fam_ids):
        idx = np.where(fam_ids == f)[0]
        if len(idx) >= 2:
            for _ in range(20):
                q = rng.permutation(idx)
                if not np.any(q == idx):
                    p[idx] = q; break
    return p


# ── features ───────────────────────────────────────────────────────────────────
def feats_pos(m, seqs, bs=128, npos=NPOS):
    out = []
    for i in range(0, len(seqs), bs):
        x = batch_onehot(seqs[i:i + bs], device=m.device)
        with torch.no_grad():
            h = F.adaptive_avg_pool1d(m.body_feats(x), npos)
        out.append(h.transpose(1, 2).cpu())
    return torch.cat(out)


def pooled(Fp):
    return torch.cat([Fp.mean(1), Fp.amax(1)], dim=1)


# ── RNA-only baselines (real body + random-body leakage control) ────────────────
class RandomBody(nn.Module):
    """Frozen randomly-initialized 2-conv body matching PARNET body shape; the control whose gap to the real
    RNA-only baseline = the PARNET-leakage-attributable share."""
    def __init__(self, d=512, k=15, device="cpu"):
        super().__init__()
        torch.manual_seed(0)
        self.c1 = nn.Conv1d(4, d, k, padding="same"); self.c2 = nn.Conv1d(d, d, k, padding="same")
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device).eval()

    @torch.no_grad()
    def body_feats(self, x):
        return torch.relu(self.c2(torch.relu(self.c1(x.float()))))

    def pooled(self, seqs, dev, bs=128):
        out = []
        for i in range(0, len(seqs), bs):
            h = self.body_feats(batch_onehot(seqs[i:i + bs], device=dev))
            out.append(torch.cat([h.mean(2), h.amax(2)], 1).cpu())
        return torch.cat(out).to(dev)


def auprc_cols(pred, te_y, ti_keep, K):
    out = np.full(K, np.nan)
    for k in range(K):
        y = (te_y[:, ti_keep[k]] > 0).astype(float)
        if y.sum() >= 5:
            out[k] = average_precision(pred[:, k], y)
    return out


def train_rna_only_multi(Ptr, Ytr, Pte, te_y, ti_keep, K, dev, seed, epochs=EPOCHS, lr=LR):
    """Track-aware multitask head on pooled features, NO protein — the fair RNA-only baseline. Equal budget."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    pw = ((len(Ytr) - Ytr.sum(0)) / (Ytr.sum(0) + 1e-6)).clamp(1, 200)
    head = nn.Sequential(nn.Linear(Ptr.shape[1], 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, K)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        perm = rng.permutation(len(Ptr))
        for i in range(0, len(perm), 512):
            b = torch.tensor(perm[i:i + 512], device=dev)
            opt.zero_grad(); lossf(head(Ptr[b]), Ytr[b]).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        return auprc_cols(torch.sigmoid(head(Pte)).cpu().numpy(), te_y, ti_keep, K)


# ── statistics ───────────────────────────────────────────────────────────────────
def boot_ci(diff, n=2000):
    rng = np.random.default_rng(); d = diff[~np.isnan(diff)]
    if len(d) == 0:
        return [float("nan"), float("nan")]
    bs = [np.mean(rng.choice(d, len(d), True)) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def sign_test(diff):
    d = diff[~np.isnan(diff)]; n = len(d); npos = int((d > 0).sum())
    bp = wp = float("nan")
    try:
        from scipy import stats as ss
        bp = float(ss.binomtest(npos, n, 0.5).pvalue) if n else float("nan")
        wp = float(ss.wilcoxon(d).pvalue) if n >= 6 and np.any(d != 0) else float("nan")
    except Exception:
        pass
    return npos, n, bp, wp


def summarize_method(REAL, SHUF, rna_multi, syms, fam_lab, ti_keep, te_y):
    """REAL/SHUF: (seeds,K). Returns the standard binding_fair-schema method record (deltas + CI + sign test)."""
    rm = np.nanmean(REAL, 0); sm = np.nanmean(SHUF, 0)
    d_rna = rm - rna_multi; valid = ~np.isnan(rm) & ~np.isnan(rna_multi)
    ci_rna = boot_ci(d_rna[valid]); ci_shuf = boot_ci((rm - sm)[~np.isnan(rm) & ~np.isnan(sm)])
    npos, nR, bp, wp = sign_test(d_rna[valid])
    rows = [{"rbp": syms[i], "family": fam_lab[i], "real": float(rm[i]), "shuffle": float(sm[i]),
             "rna_multi": float(rna_multi[i]), "vs_rna": float(d_rna[i]), "vs_shuf": float(rm[i] - sm[i])}
            for i in range(len(syms)) if valid[i]]
    return {"real": float(np.nanmean(rm)), "shuffle": float(np.nanmean(sm)),
            "gap_vs_shuffle": float(np.nanmean((rm - sm)[~np.isnan(rm) & ~np.isnan(sm)])), "gap_vs_shuffle_ci": ci_shuf,
            "gap_vs_rna_only": float(np.nanmean(d_rna[valid])), "gap_vs_rna_only_ci": ci_rna,
            "n_beat_rna_only": npos, "n_rbp": nR, "sign_test_binom_p": bp, "wilcoxon_p": wp,
            "direction_vs_rna_only": ("BEATS" if npos > nR / 2 else "UNDERPERFORMS"), "rows": rows}
