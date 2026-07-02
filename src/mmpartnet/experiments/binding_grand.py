"""COMPREHENSIVE pre-(leave-out-PARNET) check: architecture x loss/signal, with the full permutation-control
battery, stratified by family + binding strength, across binding schemes. One harness, many cells, all on
identical CV splits/seeds so every comparison is controlled.

ARCHITECTURES : film (ConditionedHead), xattn (CrossAttnHead), perres (BiCrossAttnHead)
LOSS/SIGNAL   : per cell = (neg-mode, reg, protein-signal)
   neg-mode : rand  = random-protein negative on the same window (label = does that protein also bind it)
              hard  = a protein that does NOT bind the window, family-matched when possible (anti-bypass-hard)
   reg      : none | nec = differentiable necessity gate (margin: real protein must beat a permuted one on
              positives) -- the trainable form of the protein-permutation control (from m1_lejepa necessity)
   signal   : esm | esm+string (concat STRING-PE PPI embedding) | perres (per-residue, perres arch only)
CONTROLS (eval): real ; shuffle-derangement (no fixed points -- fixes the audited bias) ; shuffle-within-family
   (permute protein only within its ATtRACT family -> tests SPECIFIC-protein vs mere family-identity)
STRATIFY (post): by family (KH/RRM/KH;RRM/other) and by binding-strength tertile; mix_coeff tertile if cached.

  python -m mmpartnet.experiments.binding_grand [scheme] [k_seeds] [n_train] [n_test]
"""
from __future__ import annotations
import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from mmpartnet import config
from mmpartnet.io import embeddings
from mmpartnet.models.heads import ConditionedHead
from mmpartnet.models.early_fusion import EarlyFusion
from mmpartnet.models.cross_attn_head import CrossAttnHead, BiCrossAttnHead
from mmpartnet.models.parnet import load_parnet
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.experiments.binding_eval import average_precision, read_rbp
from mmpartnet.experiments.binding_head import load_split

LWIN = 600
NPOS = 64
LMAX = 384
EPOCHS = 12
NEC_MARGIN = 1.0
NEC_W = 0.5


# ---- permutation controls -----------------------------------------------------
def derangement(K, rng):
    while True:
        p = rng.permutation(K)
        if not np.any(p == np.arange(K)):
            return p


def within_family_perm(fam_ids, rng):
    """Permute indices only within each family group; derange groups of size>=2, fix singletons."""
    K = len(fam_ids); p = np.arange(K)
    for f in set(fam_ids):
        idx = np.where(fam_ids == f)[0]
        if len(idx) >= 2:
            sub = idx.copy()
            for _ in range(20):
                q = rng.permutation(idx)
                if not np.any(q == idx):
                    sub = q; break
            p[idx] = sub
    return p


# ---- features -----------------------------------------------------------------
def feats_pos(m, seqs, bs=128):
    out = []
    for i in range(0, len(seqs), bs):
        x = batch_onehot(seqs[i:i + bs], device=m.device)
        with torch.no_grad():
            h = F.adaptive_avg_pool1d(m.body_feats(x), NPOS)
        out.append(h.transpose(1, 2).cpu())
    return torch.cat(out)


def make_head(arch, dp, dp_res, dev):
    if arch == "film":
        return ConditionedHead(dr=1024, dp=dp, residual=True).to(dev)
    if arch == "xattn":
        return CrossAttnHead(d_model=512, dp=dp, heads=4, layers=2).to(dev)
    return BiCrossAttnHead(d_model=512, dp=dp_res, heads=4, layers=2).to(dev)


def logit(arch, head, Xp, Xpos, E, Pres, Mres, kidx):
    """Score windows against protein column(s) kidx (LongTensor, per-row protein index)."""
    if arch == "film":
        return head(Xp, E[kidx])[0]
    if arch == "xattn":
        return head(Xpos, E[kidx])
    return head(Xpos, Pres[kidx], Mres[kidx])


def train_cell(arch, neg, reg, Xp_tr, Xpos_tr, Ytr, E, Pres, Mres, K, fam_ids, bind_by_k, seed, dev):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    head = make_head(arch, E.shape[1], Pres.shape[2], dev)
    opt = torch.optim.Adam(head.parameters(), lr=(1e-3 if arch == "film" else 5e-4), weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    pos = torch.nonzero(Ytr > 0, as_tuple=False).cpu().numpy()
    Ynp = Ytr.cpu().numpy()
    for ep in range(EPOCHS):
        rng.shuffle(pos)
        for i in range(0, len(pos), 128):
            pb = pos[i:i + 128]; nb = len(pb)
            wp = torch.tensor(pb[:, 0], device=dev); kp = torch.tensor(pb[:, 1], device=dev)
            if neg == "hard":
                kn = np.empty(nb, np.int64)
                for j, (w, k) in enumerate(pb):
                    cand = np.where((Ynp[w] == 0))[0]                       # proteins that do NOT bind w
                    fam = fam_ids[k]; same = cand[fam_ids[cand] == fam]
                    pool = same if len(same) else cand
                    kn[j] = pool[rng.integers(len(pool))] if len(pool) else k
                kn = torch.tensor(kn, device=dev)
                yneg = torch.zeros(nb, device=dev)
            else:
                kn = torch.tensor(rng.integers(0, K, size=nb), device=dev)
                yneg = Ytr[wp, kn]
            Xp2 = None if Xp_tr is None else torch.cat([Xp_tr[wp], Xp_tr[wp]])
            Xpos2 = torch.cat([Xpos_tr[wp], Xpos_tr[wp]])
            kk = torch.cat([kp, kn]); yy = torch.cat([torch.ones(nb, device=dev), yneg])
            l = logit(arch, head, Xp2, Xpos2, E, Pres, Mres, kk)
            loss = bce(l, yy)
            if reg == "nec":                                               # real protein must beat a permuted one on positives
                kperm = kp[torch.randperm(nb, device=dev)]
                lr = logit(arch, head, (Xp_tr[wp] if Xp_tr is not None else None), Xpos_tr[wp], E, Pres, Mres, kp)
                lq = logit(arch, head, (Xp_tr[wp] if Xp_tr is not None else None), Xpos_tr[wp], E, Pres, Mres, kperm)
                loss = loss + NEC_W * torch.relu(NEC_MARGIN - (lr - lq)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def eval_aps(arch, head, Xp_te, Xpos_te, te_y, ti_keep, E, Pres, Mres, K, kmap, dev):
    out = np.full(K, np.nan)
    with torch.no_grad():
        for k in range(K):
            j = int(kmap[k]); ss = []
            n = len(Xpos_te)
            for i in range(0, n, 512):
                Xp = None if Xp_te is None else Xp_te[i:i + 512]
                Xpos = Xpos_te[i:i + 512]
                idx = torch.full((len(Xpos),), j, device=dev, dtype=torch.long)
                ss.append(torch.sigmoid(logit(arch, head, Xp, Xpos, E, Pres, Mres, idx)).cpu().numpy())
            s = np.concatenate(ss); y = (te_y[:, ti_keep[k]] > 0).astype(float)
            if y.sum() >= 5:
                out[k] = average_precision(s, y)
    return out


def run_cell(arch, neg, reg, Xp_tr, Xpos_tr, Ytr, Xp_te, Xpos_te, te_y, ti_keep, E, Pres, Mres, K,
             fam_ids, bind_by_k, kseeds, dev):
    REAL = np.full((kseeds, K), np.nan); DER = np.full((kseeds, K), np.nan); FAM = np.full((kseeds, K), np.nan)
    for s in range(kseeds):
        rng = np.random.default_rng(1000 + s)
        head = train_cell(arch, neg, reg, Xp_tr, Xpos_tr, Ytr, E, Pres, Mres, K, fam_ids, bind_by_k, s, dev)
        REAL[s] = eval_aps(arch, head, Xp_te, Xpos_te, te_y, ti_keep, E, Pres, Mres, K, np.arange(K), dev)
        DER[s] = eval_aps(arch, head, Xp_te, Xpos_te, te_y, ti_keep, E, Pres, Mres, K, derangement(K, rng), dev)
        FAM[s] = eval_aps(arch, head, Xp_te, Xpos_te, te_y, ti_keep, E, Pres, Mres, K, within_family_perm(fam_ids, rng), dev)
    rm = np.nanmean(REAL, 0); dm = np.nanmean(DER, 0); fm = np.nanmean(FAM, 0)
    valid = ~np.isnan(rm)
    gap_der = (rm - dm)[valid]; gap_fam = (rm - fm)[valid]
    rng = np.random.default_rng(0)
    ci = lambda d: (float(np.percentile([np.mean(rng.choice(d, len(d), True)) for _ in range(1500)], 2.5)),
                    float(np.percentile([np.mean(rng.choice(d, len(d), True)) for _ in range(1500)], 97.5)))
    return {"real": float(np.nanmean(rm)), "shuf_derange": float(np.nanmean(dm)), "shuf_family": float(np.nanmean(fm)),
            "gap_derange": float(np.nanmean(gap_der)), "gap_family": float(np.nanmean(gap_fam)),
            "gap_der_ci": ci(gap_der), "n_rbp": int(valid.sum()),
            "rm": rm.tolist(), "dm": dm.tolist(), "fm": fm.tolist()}


def stratify(rm, dm, fm, fam_lab, valid):
    """Mean real and real-derange gap within family groups and binding-strength tertiles."""
    rm = np.array(rm); dm = np.array(dm); fm = np.array(fm)
    out = {"by_family": {}, "by_strength": {}}
    for f in sorted(set(fam_lab)):
        m = (np.array(fam_lab) == f) & valid
        if m.sum() >= 3:
            out["by_family"][f] = {"n": int(m.sum()), "real": float(np.nanmean(rm[m])),
                                   "gap_derange": float(np.nanmean((rm - dm)[m])),
                                   "gap_family": float(np.nanmean((rm - fm)[m]))}
    order = np.argsort(np.where(valid, rm, -1))
    vidx = order[~np.isnan(rm[order])]
    vidx = np.array([i for i in order if valid[i]])
    thirds = np.array_split(vidx, 3)
    for name, idx in zip(["low", "mid", "high"], thirds):
        if len(idx):
            out["by_strength"][name] = {"n": int(len(idx)), "real": float(np.nanmean(rm[idx])),
                                        "gap_derange": float(np.nanmean((rm - dm)[idx]))}
    return out


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "pureclip"
    kseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 25000
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 15000
    panel = sys.argv[5] if len(sys.argv) > 5 else "core"     # core = esm n perres (11 cells); full = esm-only scaling (film/xattn x esm/string)
    base = os.path.expanduser(f"~/ml4rg_data/binding/{scheme}")
    bundle = os.environ.get("ML4RG_DATA", os.path.expanduser("~/ml4rg_data"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(f"{base}/dataset.pt"):
        print(f"SCHEME {scheme} not found at {base}; abort"); return
    tracks = read_rbp(f"{base}/rbp_cts.tsv"); pr = embeddings.ProteinRep()
    z = np.load(f"{bundle}/perres32.npz", allow_pickle=True); have = set(z.files)
    if panel == "full":
        keep = [(ti, r, c) for ti, (r, c) in enumerate(tracks) if pr.esm(r) is not None]
    else:
        keep = [(ti, r, c) for ti, (r, c) in enumerate(tracks) if pr.esm(r) is not None and r in have]
    K = len(keep); ti_keep = [ti for ti, r, c in keep]; syms = [r for ti, r, c in keep]
    m = load_parnet(device=dev)

    # family map (ATtRACT cohort); STRING-PE coverage
    from mmpartnet.io import cohort
    fam_map = cohort.attract_families(syms)
    fam_lab = [(fam_map[r][0] if r in fam_map and fam_map[r] else "other") for r in syms]
    fam_ids = np.array([sorted(set(fam_lab)).index(f) for f in fam_lab])
    n_string = sum(pr.string_pe(r) is not None for r in syms)
    print(f"binding_grand [{scheme}] K={K} seeds={kseeds} dev={dev} | STRING cover {n_string}/{K} | "
          f"families {len(set(fam_lab))}", flush=True)

    tr_seq, tr_y = load_split(base, "train", n_train); te_seq, te_y = load_split(base, "test", n_test)
    print(f"feats: train={len(tr_seq)} test={len(te_seq)} ...", flush=True)
    Ftr = feats_pos(m, tr_seq).to(dev); Fte = feats_pos(m, te_seq).to(dev)
    Ptr = torch.cat([Ftr.mean(1), Ftr.amax(1)], 1); Pte = torch.cat([Fte.mean(1), Fte.amax(1)], 1)
    Eesm = torch.stack([torch.tensor(pr.esm(r), dtype=torch.float32) for r in syms]).to(dev)
    sdim = next((len(pr.string_pe(r)) for r in syms if pr.string_pe(r) is not None), 0)
    Estr = torch.stack([torch.tensor(np.concatenate([pr.esm(r), (pr.string_pe(r) if pr.string_pe(r) is not None else np.zeros(sdim))]), dtype=torch.float32) for r in syms]).to(dev) if sdim else Eesm
    PRESn = np.zeros((K, LMAX, 32), np.float32); MASKn = np.ones((K, LMAX), bool)
    for i, s in enumerate(syms):
        a = np.asarray(z[s], np.float32)[:LMAX]; PRESn[i, :len(a)] = a; MASKn[i, :len(a)] = False
    Pres = torch.tensor(PRESn).to(dev); Mres = torch.tensor(MASKn).to(dev)
    Ytr = torch.tensor((tr_y[:, ti_keep] > 0).astype(np.float32)).to(dev)
    bind_by_k = None

    # cells: (arch, neg, reg, signal). full-panel scaling drops perres (no perres for many RBPs) + the
    # already-characterized hard/nec, keeping the two scaling questions: does the ladder + STRING hold at scale.
    if panel == "full":
        cells = [("film", "rand", "none", "esm"), ("film", "rand", "none", "string"),
                 ("xattn", "rand", "none", "esm"), ("xattn", "rand", "none", "string")]
    else:
        cells = [
            ("film", "rand", "none", "esm"), ("film", "hard", "none", "esm"), ("film", "rand", "nec", "esm"), ("film", "rand", "none", "string"),
            ("xattn", "rand", "none", "esm"), ("xattn", "hard", "none", "esm"), ("xattn", "rand", "nec", "esm"), ("xattn", "rand", "none", "string"),
            ("perres", "rand", "none", "esm"), ("perres", "hard", "none", "esm"), ("perres", "rand", "nec", "esm"),
        ]
    results = {}
    for arch, neg, reg, sig in cells:
        name = f"{arch}/{neg}/{reg}/{sig}"
        E = Estr if sig == "string" else Eesm
        Xp_tr, Xp_te = (Ptr, Pte) if arch == "film" else (None, None)
        res = run_cell(arch, neg, reg, Xp_tr, Ftr, Ytr, Xp_te, Fte, te_y, ti_keep, E, Pres, Mres, K, fam_ids, bind_by_k, kseeds, dev)
        valid = ~np.isnan(np.array(res["rm"]))
        res["strata"] = stratify(res["rm"], res["dm"], res["fm"], fam_lab, valid)
        results[name] = res
        print(f"  {name:24} real {res['real']:.3f}  gap_der {res['gap_derange']:+.3f}  gap_fam {res['gap_family']:+.3f}  "
              f"CI[{res['gap_der_ci'][0]:+.3f},{res['gap_der_ci'][1]:+.3f}]", flush=True)

    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    tag = f"_{panel}" if panel != "core" else ""
    (o / f"binding_grand_{scheme}{tag}.json").write_text(json.dumps(
        {"scheme": scheme, "panel": panel, "seeds": kseeds, "K": K, "n_string": n_string, "rbps": syms,
         "fam_lab": fam_lab, "cells": results}, indent=1))
    print(f"wrote binding_grand_{scheme}{tag}.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
