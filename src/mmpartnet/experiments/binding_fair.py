"""FAIR controlled comparison of every team's conditioning method vs proper baselines, on ONE harness:
identical lab substrate, identical RBP panel (ESM ∩ ProtT5 ∩ per-residue ∩ STRING), identical split, seeds,
negatives, optimizer budget, and metric (per-RBP auPRC). This makes Christoph's early-fusion, our FiLM /
cross-attention / per-residue heads, and the baselines directly comparable — and answers "does conditioning
beat an RNA-only baseline", not just "does the specific protein matter".

Baselines:
  parnet_zeroshot   frozen PARNET pretrained per-track output, pooled (no training)        [floor]
  rna_only_multi    track-aware multitask head on PARNET pooled feats, NO protein           [strong RNA-only]
  rna_only_bind     the FiLM head's protein-agnostic rna_only branch (single bindability)   [protein-agnostic null]
Methods (protein-conditioned, identical feats/negs/epochs):
  concat            EarlyFusion concat[PARNET pooled, protein]  (= Christoph's early-fusion on PARNET feats)
  film              FiLM ConditionedHead
  xattn             cross-attention head (protein query x RNA positions)
  perres            per-residue BiCrossAttn (RNA positions x protein residues)
Each method is scored REAL and under a derangement protein-SHUFFLE; we report, per RBP, the method auPRC and
its deltas vs (a) rna_only_multi [the fair "beats RNA-only?" test] and (b) shuffle [specificity], with
bootstrap CIs + paired sign tests.

NOTE: in-distribution (the all-223 PARNET body saw these RBPs) the track-aware rna_only_multi is a STRONG
baseline; the decisive test is the same harness under the leave-out PARNET (swap ML4RG_PARNET_WEIGHTS).

  python -m mmpartnet.experiments.binding_fair [scheme] [k_seeds] [n_train] [n_test]
"""
from __future__ import annotations
import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from mmpartnet import config
from mmpartnet.io import embeddings, cohort
from mmpartnet.models.early_fusion import EarlyFusion
from mmpartnet.models.heads import ConditionedHead
from mmpartnet.models.cross_attn_head import CrossAttnHead, BiCrossAttnHead
from mmpartnet.models.parnet import load_parnet
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.experiments.binding_eval import average_precision, read_rbp
from mmpartnet.experiments.binding_head import load_split
from mmpartnet.experiments.binding_grand import feats_pos, derangement, within_family_perm, LMAX

EPOCHS = 12
LR = 5e-4            # equalized across ALL arms (fairness fix)


class RandomBody(nn.Module):
    """Frozen randomly-initialized 2-conv body (matches PARNET body shape) — the control that quantifies how
    much of the RNA-only baseline is real RBP structure vs all-223 PARNET leakage."""
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


def auprc_cols(pred, te_y, ti_keep, K):
    out = np.full(K, np.nan)
    for k in range(K):
        y = (te_y[:, ti_keep[k]] > 0).astype(float)
        if y.sum() >= 5:
            out[k] = average_precision(pred[:, k], y)
    return out


# ---- baselines -------------------------------------------------------------------
def train_rna_only_multi(Ptr, Ytr, Pte, te_y, ti_keep, K, dev, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    pw = ((len(Ytr) - Ytr.sum(0)) / (Ytr.sum(0) + 1e-6)).clamp(1, 200)
    head = nn.Sequential(nn.Linear(Ptr.shape[1], 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, K)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(EPOCHS):
        perm = rng.permutation(len(Ptr))
        for i in range(0, len(perm), 512):
            b = torch.tensor(perm[i:i + 512], device=dev)
            opt.zero_grad(); lossf(head(Ptr[b]), Ytr[b]).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pred = torch.sigmoid(head(Pte)).cpu().numpy()
    return auprc_cols(pred, te_y, ti_keep, K)


def parnet_zeroshot(m, te_seq, te_y, keep, dev, bs=128):
    """Frozen PARNET pretrained per-track total, pooled over the window -> binding score per RBP."""
    sel = []
    for ti, r, c in keep:
        t = m.track_index(r, c) if hasattr(m, "track_index") else None
        sel.append(t)
    if any(s is None for s in sel):
        return None
    scores = np.zeros((len(te_seq), len(keep)), np.float32)
    with torch.no_grad():
        for i in range(0, len(te_seq), bs):
            chunk = te_seq[i:i + bs]
            tot = m.full(batch_onehot(chunk, device=dev))["total"]    # (b, 223, L)
            pooled = tot.amax(dim=2).cpu().numpy()                    # peak track signal
            for k, t in enumerate(sel):
                scores[i:i + len(chunk), k] = pooled[:, t]
    ti_keep = list(range(len(keep)))
    # auPRC vs each track's labels (te_y aligned to keep order via ti_keep_full)
    return scores


# ---- conditioned methods ----------------------------------------------------------
def train_conditioned(kind, Ftr, Ptr, Ytr, E, PRES, MASK, K, dev, seed):
    """Returns (head, kind-tag). Trained with positives + random-protein negatives, identical budget."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    if kind == "concat":
        head = EarlyFusion(dr=Ptr.shape[1], dp=E.shape[1]).to(dev)
    elif kind == "film":
        head = ConditionedHead(dr=Ptr.shape[1], dp=E.shape[1], residual=True).to(dev)
    elif kind == "xattn":
        head = CrossAttnHead(d_model=Ftr.shape[2], dp=E.shape[1], heads=4, layers=2).to(dev)
    else:
        head = BiCrossAttnHead(d_model=Ftr.shape[2], dp=PRES.shape[2], heads=4, layers=2).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    pos = torch.nonzero(Ytr > 0, as_tuple=False).cpu().numpy()

    def fwd(Xp, Xpos, idx):
        if kind == "concat":
            return head(Xp, E[idx])
        if kind == "film":
            return head(Xp, E[idx])[0]
        if kind == "xattn":
            return head(Xpos, E[idx])
        return head(Xpos, PRES[idx], MASK[idx])

    for ep in range(EPOCHS):
        rng.shuffle(pos)
        for i in range(0, len(pos), 128):
            pb = pos[i:i + 128]; nb = len(pb)
            wp = torch.tensor(pb[:, 0], device=dev); kp = torch.tensor(pb[:, 1], device=dev)
            kn = torch.tensor(rng.integers(0, K, size=nb), device=dev)
            kk = torch.cat([kp, kn]); yy = torch.cat([torch.ones(nb, device=dev), Ytr[wp, kn]])
            Xp2 = torch.cat([Ptr[wp], Ptr[wp]]) if kind in ("concat", "film") else None
            Xpos2 = torch.cat([Ftr[wp], Ftr[wp]]) if kind in ("xattn", "perres") else None
            opt.zero_grad(); bce(fwd(Xp2, Xpos2, kk), yy).backward(); opt.step()
    head.eval()
    return head, fwd


def eval_conditioned(kind, head, fwd, Fte, Pte, te_y, ti_keep, K, kmap, dev):
    out = np.full(K, np.nan)
    with torch.no_grad():
        for k in range(K):
            j = int(kmap[k]); ss = []
            n = len(Fte)
            for i in range(0, n, 256):
                idx = torch.full((min(256, n - i),), j, device=dev, dtype=torch.long)
                Xp = Pte[i:i + 256] if kind in ("concat", "film") else None
                Xpos = Fte[i:i + 256] if kind in ("xattn", "perres") else None
                ss.append(torch.sigmoid(fwd(Xp, Xpos, idx)).cpu().numpy())
            s = np.concatenate(ss); y = (te_y[:, ti_keep[k]] > 0).astype(float)
            if y.sum() >= 5:
                out[k] = average_precision(s, y)
    return out


def eval_film_rnaonly(head, Pte, te_y, ti_keep, K, dev):
    """The FiLM head's protein-agnostic rna_only branch b (same bindability for every RBP)."""
    with torch.no_grad():
        e0 = torch.zeros(1, head.film.in_features, device=dev)
        bs = []
        for i in range(0, len(Pte), 256):
            _, b, _ = head(Pte[i:i + 256], e0.expand(min(256, len(Pte) - i), -1))
            bs.append(torch.sigmoid(b).cpu().numpy())
    b = np.concatenate(bs)
    out = np.full(K, np.nan)
    for k in range(K):
        y = (te_y[:, ti_keep[k]] > 0).astype(float)
        if y.sum() >= 5:
            out[k] = average_precision(b, y)
    return out


def summarize(per_seed):  # per_seed: (S,K) -> mean over seeds
    return np.nanmean(per_seed, 0)


def boot_ci(diff, n=2000):
    """Paired bootstrap (resamples the K per-RBP paired deltas) with a LIVE RNG (no fixed seed -> does not
    understate Monte-Carlo noise)."""
    rng = np.random.default_rng(); d = diff[~np.isnan(diff)]
    if len(d) == 0:
        return [float("nan"), float("nan")]
    bs = [np.mean(rng.choice(d, len(d), True)) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def sign_test(diff):
    """Binomial sign test on per-RBP paired deltas vs H0 p=0.5 + Wilcoxon signed-rank. Returns
    (n_pos, n, binom_p, wilcoxon_p). Reports DIRECTION honestly (n_pos < n/2 => method underperforms)."""
    d = diff[~np.isnan(diff)]; n = len(d); npos = int((d > 0).sum())
    bp = wp = float("nan")
    try:
        from scipy import stats as ss
        bp = float(ss.binomtest(npos, n, 0.5).pvalue) if n else float("nan")
        wp = float(ss.wilcoxon(d).pvalue) if n >= 6 and np.any(d != 0) else float("nan")
    except Exception:
        pass
    return npos, n, bp, wp


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "pureclip"
    kseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 25000
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 15000
    base = os.path.expanduser(f"~/ml4rg_data/binding/{scheme}")
    bundle = os.environ.get("ML4RG_DATA", os.path.expanduser("~/ml4rg_data"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = read_rbp(f"{base}/rbp_cts.tsv"); pr = embeddings.ProteinRep()
    z = np.load(f"{bundle}/perres32.npz", allow_pickle=True); have = set(z.files)
    p5p = os.path.expanduser("~/ml4rg_data/prott5_reduced.npz")
    p5 = np.load(p5p) if os.path.exists(p5p) else None; p5set = set(p5.files) if p5 is not None else set()
    keep = [(ti, r, c) for ti, (r, c) in enumerate(tracks)
            if pr.esm(r) is not None and r in have and pr.string_pe(r) is not None and r in p5set]
    K = len(keep); ti_keep = [ti for ti, r, c in keep]; syms = [r for ti, r, c in keep]
    fam_map = cohort.attract_families(syms)
    fam_lab = [(fam_map[r][0] if r in fam_map and fam_map[r] else "other") for r in syms]
    m = load_parnet(device=dev)
    print(f"binding_fair [{scheme}] common-panel K={K} seeds={kseeds} train={n_train} test={n_test} dev={dev}", flush=True)
    tr_seq, tr_y = load_split(base, "train", n_train); te_seq, te_y = load_split(base, "test", n_test)
    Ftr = feats_pos(m, tr_seq).to(dev); Fte = feats_pos(m, te_seq).to(dev)
    Ptr = torch.cat([Ftr.mean(1), Ftr.amax(1)], 1); Pte = torch.cat([Fte.mean(1), Fte.amax(1)], 1)
    Ytr = torch.tensor((tr_y[:, ti_keep] > 0).astype(np.float32)).to(dev)
    E = torch.stack([torch.tensor(pr.esm(r), dtype=torch.float32) for r in syms]).to(dev)
    PRESn = np.zeros((K, LMAX, 32), np.float32); MASKn = np.ones((K, LMAX), bool)
    for i, s in enumerate(syms):
        a = np.asarray(z[s], np.float32)[:LMAX]; PRESn[i, :len(a)] = a; MASKn[i, :len(a)] = False
    PRES = torch.tensor(PRESn).to(dev); MASK = torch.tensor(MASKn).to(dev)

    # random-body control: pooled feats from a FROZEN random body (quantifies PARNET leakage in the baseline)
    rb = RandomBody(device=dev)
    def rb_pooled(seqs, bs=128):
        out = []
        for i in range(0, len(seqs), bs):
            with torch.no_grad():
                h = rb.body_feats(batch_onehot(seqs[i:i + bs], device=dev))
            out.append(torch.cat([h.mean(2), h.amax(2)], 1).cpu())
        return torch.cat(out)
    Ptr_rb = rb_pooled(tr_seq).to(dev); Pte_rb = rb_pooled(te_seq).to(dev)

    leak = "all-223 (LEAKED, in-distribution)" if "heldout" not in os.environ.get("ML4RG_PARNET_HELDOUT", "") else "leave-out"
    print(f"PARNET body training set = {leak}  (frozen RNA-only baseline is leakage-inflated under all-223)", flush=True)

    methods = ["concat", "film", "xattn", "perres"]
    REAL = {k: np.full((kseeds, K), np.nan) for k in methods}
    SHUF = {k: np.full((kseeds, K), np.nan) for k in methods}
    RNAMULTI = np.full((kseeds, K), np.nan); RNABIND = np.full((kseeds, K), np.nan); RNARAND = np.full((kseeds, K), np.nan)

    for s in range(kseeds):
        RNAMULTI[s] = train_rna_only_multi(Ptr, Ytr, Pte, te_y, ti_keep, K, dev, s)
        RNARAND[s] = train_rna_only_multi(Ptr_rb, Ytr, Pte_rb, te_y, ti_keep, K, dev, s)  # random-body control
        rngp = np.random.default_rng(1000 + s); der = derangement(K, rngp)
        for kind in methods:
            head, fwd = train_conditioned(kind, Ftr, Ptr, Ytr, E, PRES, MASK, K, dev, s)
            REAL[kind][s] = eval_conditioned(kind, head, fwd, Fte, Pte, te_y, ti_keep, K, np.arange(K), dev)
            SHUF[kind][s] = eval_conditioned(kind, head, fwd, Fte, Pte, te_y, ti_keep, K, der, dev)
            if kind == "film":
                RNABIND[s] = eval_film_rnaonly(head, Pte, te_y, ti_keep, K, dev)
        print(f"  seed {s} done", flush=True)

    rna_multi = summarize(RNAMULTI); rna_bind = summarize(RNABIND); rna_rand = summarize(RNARAND)
    results = {"rna_only_multitask": float(np.nanmean(rna_multi)),
               "rna_only_bindability": float(np.nanmean(rna_bind)),
               "rna_only_randombody": float(np.nanmean(rna_rand)), "methods": {}}
    leak_frac = float(np.nanmean(rna_multi) - np.nanmean(rna_rand))
    print(f"\nBASELINES: rna_only_multitask {np.nanmean(rna_multi):.4f}  random-body {np.nanmean(rna_rand):.4f} "
          f"(leakage-attributable {leak_frac:+.4f})  protein-agnostic-branch {np.nanmean(rna_bind):.4f}", flush=True)
    for kind in methods:
        rm = summarize(REAL[kind]); sm = summarize(SHUF[kind])
        d_rna = rm - rna_multi
        valid = ~np.isnan(rm) & ~np.isnan(rna_multi)
        ci_rna = boot_ci(d_rna[valid]); ci_shuf = boot_ci((rm - sm)[~np.isnan(rm) & ~np.isnan(sm)])
        npos, nR, binom_p, wil_p = sign_test(d_rna[valid])
        direction = "BEATS" if npos > nR / 2 else "UNDERPERFORMS"
        rows = [{"rbp": syms[i], "family": fam_lab[i], "real": float(rm[i]), "shuffle": float(sm[i]),
                 "rna_multi": float(rna_multi[i]), "vs_rna": float(d_rna[i]), "vs_shuf": float(rm[i] - sm[i])}
                for i in range(K) if valid[i]]
        results["methods"][kind] = {
            "real": float(np.nanmean(rm)), "shuffle": float(np.nanmean(sm)),
            "gap_vs_shuffle": float(np.nanmean((rm - sm)[~np.isnan(rm) & ~np.isnan(sm)])), "gap_vs_shuffle_ci": ci_shuf,
            "gap_vs_rna_only": float(np.nanmean(d_rna[valid])), "gap_vs_rna_only_ci": ci_rna,
            "n_beat_rna_only": npos, "n_rbp": nR, "sign_test_binom_p": binom_p, "wilcoxon_p": wil_p,
            "direction_vs_rna_only": direction, "rows": rows}
        r = results["methods"][kind]
        print(f"  {kind:7} real {r['real']:.4f} | vs RNA-only {r['gap_vs_rna_only']:+.4f} "
              f"CI[{ci_rna[0]:+.4f},{ci_rna[1]:+.4f}] | {direction} {npos}/{nR} (binom p={binom_p:.1e}, "
              f"wilcoxon p={wil_p:.1e}) | vs shuffle {r['gap_vs_shuffle']:+.4f}", flush=True)

    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    (o / "binding_fair.json").write_text(json.dumps(
        {"scheme": scheme, "panel": "esm∩prott5∩perres∩string", "K": K, "seeds": kseeds,
         "n_train": len(tr_seq), "n_test": len(te_seq),
         "parnet_body": leak, "in_distribution": ("leave-out" not in leak),
         "lr": LR, "epochs": EPOCHS,
         "baselines": {k: results[k] for k in ("rna_only_multitask", "rna_only_randombody", "rna_only_bindability")},
         "leakage_attributable_auprc": leak_frac,
         "methods": results["methods"], "rbps": syms}, indent=1))
    print("wrote binding_fair.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
