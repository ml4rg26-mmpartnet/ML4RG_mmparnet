"""M1 BINARY HELD-OUT-RBP GATE on the CLEAN leave-out-RBP backbone (the roadmap do-next: CORAL's pre-registered
metric -- fraction of held-out RBPs with auROC > 0.65). A protein-CONDITIONED binary classifier (ConditionedHead:
clean-PARNET window features x ESM protein -> binding logit) is trained ONLY on the RETAINED RBPs of a clean
backbone fold and evaluated ZERO-SHOT on that fold's HELD-OUT RBPs (matched BY NAME to the RBPs the clean backbone
never saw, so the gate is leakage-clean). Reports per-held-RBP auROC + auPRC, the fraction auROC>0.65, and the
real-vs-protein-shuffle and real-vs-within-family controls.

  CLEAN_BB=$HOME/jobs/mmpartnet-clean5fold/clean_backbone_clean_scratch_f{F}.pt M2_FOLD=F M2_KFOLD=5 \
  python -m mmpartnet.experiments.binding_gate [scheme] [n_train] [n_test]

The held set per fold MUST equal the clean backbone's held set: both use np.array_split(default_rng(0).permutation
(K_m2), kf)[fold] over the SAME M2 RBP name ordering (read_full_rbp_cts + _cell_select), then intersected with the
binary task's RBP set by NAME. A protein-shuffle control on the held RBPs must give auROC ~0.5 (leakage sanity).
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
from mmpartnet.models.heads import ConditionedHead
from mmpartnet.models.parnet import load_parnet
from mmpartnet.experiments.binding_eval import average_precision, read_rbp
from mmpartnet.experiments.binding_head import feats, load_split


def roc_auc(score, y):
    """rank-based AUROC (no sklearn dep). y in {0,1}."""
    y = np.asarray(y); s = np.asarray(score, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    return float((ranks[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def m2_held_names(kf, fold):
    """The RBP NAMES the clean backbone for (kf,fold) held out -- recomputed from the M2 RBP ordering so the
    binary gate's held set matches the backbone EXACTLY (else leakage)."""
    pr = embeddings.ProteinRep()
    cell = os.environ.get("M2_CELL", "HepG2")
    from mmpartnet.data.sources.hfds import read_full_rbp_cts, _cell_select
    bundle = os.environ.get("ML4RG_DATA", os.path.expanduser("~/ml4rg_data"))
    z = np.load(f"{bundle}/perres32.npz", allow_pickle=True); have = set(z.files)
    cts = read_full_rbp_cts()
    syms0 = [r["rbp"] for r in cts if r["ct"] == cell and pr.esm(r["rbp"]) is not None and r["rbp"] in have]
    seen = set(); syms0 = [s for s in syms0 if not (s in seen or seen.add(s))]
    _loc, syms_w, _sy, _fu = _cell_select(syms0, cts, cell)
    K = len(syms_w); perm = np.random.default_rng(0).permutation(K)
    held = set(np.array_split(perm, kf)[fold % kf].tolist())
    return set(str(syms_w[i]) for i in held)


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "pureclip"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    kf = int(os.environ.get("M2_KFOLD", "5")); fold = int(os.environ.get("M2_FOLD", "0"))
    bar = float(os.environ.get("M1_BAR", "0.65"))
    base = os.path.expanduser(f"~/ml4rg_data/binding/{scheme}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = read_rbp(f"{base}/rbp_cts.tsv"); pr = embeddings.ProteinRep()
    keep = [(ti, r, c) for ti, (r, c) in enumerate(tracks) if pr.esm(r) is not None]
    K = len(keep); syms = [r for ti, r, c in keep]
    try:
        fam_map = cohort.attract_families(syms)
        fam_lab = [(fam_map[r][0] if r in fam_map and fam_map[r] else "other") for r in syms]
    except Exception as e:                                    # ATtRACT/pwm assets absent -> drop within-family control
        print(f"[warn] family control unavailable ({type(e).__name__}); within-family = 'other'", flush=True)
        fam_lab = ["other"] * K

    if os.environ.get("M1_INDIST", "0") == "1":
        # IN-DISTRIBUTION mode (reconcile Christoph's early-fusion result): NO RBP holdout -- train on ALL RBPs,
        # eval ALL RBPs on the held-out WINDOW split (test set). Here the protein acts as an RBP-identity signal
        # the head can memorize, so protein-shuffle SHOULD break it (Christoph: 0.83->0.55). Contrast with the
        # zero-shot leave-out-RBP gate, where the protein must GENERALIZE to unseen RBPs (and does not, at binary).
        held_k = list(range(K)); train_k = list(range(K)); held_names = set(syms)
        print(f"M1 GATE [{scheme}] IN-DIST (no RBP holdout, window split): train+eval all K={K} RBPs "
              f"(backbone={'clean' if os.environ.get('CLEAN_BB') else 'leaked-all223'})", flush=True)
    elif os.environ.get("M1_SELFHELD", "0") == "1":
        # FULL-SCALE self-held mode: plain k-fold over the binary RBP set (no clean-backbone matching).
        # Runs on the provided (here leaked all-223) backbone; the protein-SHUFFLE control is the leakage
        # gate (must stay ~0.5). Held RBPs are still excluded from HEAD training, so the conditioned head
        # predicts an unseen RBP purely from its protein rep.
        # hold out by UNIQUE RBP NAME (not track): same RBP can appear in HepG2+K562 -> a per-track
        # holdout would leak the held protein via its other-cell track. Fold over unique names.
        uniq = sorted(set(syms))
        permn = np.random.default_rng(0).permutation(len(uniq))
        held_names = set(uniq[i] for i in np.array_split(permn, kf)[fold % kf].tolist())
        held_k = [k for k in range(K) if syms[k] in held_names]
        train_k = [k for k in range(K) if syms[k] not in held_names]
        print(f"M1 GATE [{scheme}] SELF-HELD fold {fold}/{kf}: {len(held_names)} held RBP names -> "
              f"held {len(held_k)} / train {len(train_k)} tracks of K={K} "
              f"(backbone={'clean' if os.environ.get('CLEAN_BB') else 'leaked-all223'})", flush=True)
    else:
        held_names = m2_held_names(kf, fold)                     # the clean backbone's held set (by name)
        held_k = [k for k in range(K) if syms[k] in held_names]  # held RBPs present in the binary task
        train_k = [k for k in range(K) if syms[k] not in held_names]
        print(f"M1 GATE [{scheme}] fold {fold}/{kf}: clean-backbone held names={sorted(held_names)} -> "
              f"held in binary set={len(held_k)} train={len(train_k)} of K={K}", flush=True)
    if len(held_k) < 3:
        print("WARN <3 held RBPs in the binary set for this fold; gate underpowered", flush=True)

    m = load_parnet(device=dev)
    cb = os.environ.get("CLEAN_BB", "")
    if cb and os.path.exists(cb):
        m.m.load_state_dict(torch.load(cb, map_location=dev)); m.m.eval()
        print(f"[clean-backbone] loaded {os.path.basename(cb)} (leakage-clean for the held set)", flush=True)
    else:
        print("[clean-backbone] WARNING: no CLEAN_BB -> running on the LEAKED backbone (not a clean gate)", flush=True)

    tr_seq, tr_y = load_split(base, "train", n_train); te_seq, te_y = load_split(base, "test", n_test)
    Ftr = feats(m, tr_seq).to(dev); Fte = feats(m, te_seq).to(dev)
    ti_keep = [ti for ti, r, c in keep]
    E = torch.stack([torch.tensor(pr.esm(r), dtype=torch.float32) for ti, r, c in keep]).to(dev)
    Ytr = torch.tensor((tr_y[:, ti_keep] > 0).astype(np.float32)).to(dev)

    head = ConditionedHead(dr=Ftr.shape[1], dp=E.shape[1], residual=True).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    trk = np.array(train_k)
    # positives only among TRAIN RBPs (held RBPs never enter training); negatives = random TRAIN RBP
    posidx = np.array([[w, k] for w, k in torch.nonzero(Ytr > 0, as_tuple=False).cpu().numpy() if k in set(train_k)])
    print(f"train positives={len(posidx)} over {len(train_k)} retained RBPs", flush=True)
    for ep in range(12):
        rng.shuffle(posidx)
        for i in range(0, len(posidx), 256):
            pb = posidx[i:i + 256]
            wp = torch.tensor(pb[:, 0], device=dev); kp = torch.tensor(pb[:, 1], device=dev)
            kn = torch.tensor(trk[rng.integers(0, len(trk), size=len(pb))], device=dev)   # neg = random TRAIN RBP
            wf = torch.cat([Ftr[wp], Ftr[wp]]); pe = torch.cat([E[kp], E[kn]])
            y = torch.cat([torch.ones(len(pb), device=dev), Ytr[wp, kn]])
            logit, _, _ = head(wf, pe)
            opt.zero_grad(); bce(logit, y).backward(); opt.step()
    head.eval()

    def eval_held(Emap):
        rows = []
        with torch.no_grad():
            for k in held_k:
                y = (te_y[:, ti_keep[k]] > 0).astype(float)
                if y.sum() < 5:
                    continue
                s = torch.sigmoid(head(Fte, Emap[k].expand(len(Fte), -1))[0]).cpu().numpy()
                rows.append({"rbp": syms[k], "family": fam_lab[k], "auroc": roc_auc(s, y),
                             "auprc": average_precision(s, y), "pos_rate": float(y.mean())})
        return rows

    # within-family shuffle of the protein among held RBPs (harder control)
    def fam_perm(ks):
        ks = list(ks); out = {k: k for k in ks}; rng2 = np.random.default_rng(1)
        byf = {}
        for k in ks:
            byf.setdefault(fam_lab[k], []).append(k)
        for f, g in byf.items():
            if len(g) >= 2:
                p = rng2.permutation(g)
                for a, b in zip(g, p):
                    out[a] = b
        return out

    real = eval_held(E)
    shuf = eval_held({k: E[v] for k, v in zip(held_k, rng.permutation(np.array(held_k)))})
    famp = fam_perm(held_k); fam = eval_held({k: E[famp[k]] for k in held_k})

    ar = np.array([r["auroc"] for r in real]); frac = float(np.mean(ar > bar)) if len(ar) else float("nan")
    ash = np.mean([r["auroc"] for r in shuf]) if shuf else float("nan")
    afm = np.mean([r["auroc"] for r in fam]) if fam else float("nan")
    print(f"\n[M1 GATE fold {fold}] held-RBP auROC mean {np.nanmean(ar):.3f} | FRACTION>{bar} = {frac:.2f} "
          f"({int(np.nansum(ar > bar))}/{len(ar)}) | protein-shuffle {ash:.3f} | within-family {afm:.3f}", flush=True)
    print("  (leakage sanity: protein-shuffle auROC should be ~0.5; CORAL bar = >=50% of held RBPs above "
          f"{bar})", flush=True)
    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("M1_TAG", f"gate_f{fold}")
    (o / f"binding_gate_{tag}.json").write_text(json.dumps(
        {"scheme": scheme, "fold": fold, "kf": kf, "bar": bar, "clean_bb": os.path.basename(cb) if cb else None,
         "n_held": len(ar), "auroc_mean": float(np.nanmean(ar)) if len(ar) else None,
         "frac_above_bar": frac, "auroc_shuffle": float(ash), "auroc_within_family": float(afm),
         "rows": real, "rows_shuffle": shuf}, indent=1))
    print(f"wrote binding_gate_{tag}.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
