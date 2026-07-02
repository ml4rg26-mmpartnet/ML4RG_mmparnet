"""Conditioning-MECHANISM ablation on the multimodal binding task, controlled. Runs three heads that
differ ONLY in how the RNA representation is conditioned on the protein, on IDENTICAL features, splits,
seeds and negative draws -- so any auPRC difference is the mechanism, not the data (the earlier
xattn-vs-FiLM comparison used 25k vs 30k windows; this removes that confound):

  concat  EarlyFusion     concat(pooled RNA, protein) -> MLP            (no interaction; the floor)
  film    ConditionedHead protein FiLM-modulates the pooled RNA rep     (global affine modulation)
  xattn   CrossAttnHead   protein query cross-attends RNA positions     (position-selective; TFBindFormer)

All three read the SAME per-position PARNET features (B,P,512); the pooled heads use mean+max over those P
positions (B,1024), so they differ only in the conditioning block. Each head trained with its own tuned LR
(architectures are tuned separately by convention; logged). Outputs binding_mechanism.json with per-RBP
real/shuffle auPRC for each mechanism + bootstrap CI of the real-vs-shuffle gap, ready for the forest plot
and the per-RBP gain-vs-ISM-sharpness interpretability bridge.

  python -m mmpartnet.experiments.binding_mechanism [scheme] [k_seeds] [n_train] [n_test]
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
from mmpartnet.models.cross_attn_head import CrossAttnHead
from mmpartnet.models.parnet import load_parnet
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.experiments.binding_eval import average_precision, read_rbp
from mmpartnet.experiments.binding_head import load_split

LWIN = 600
NPOS = 64
LR = {"concat": 1e-3, "film": 1e-3, "xattn": 5e-4}
EPOCHS = 12


def feats_pos(m, seqs, bs=128):
    """Per-position PARNET features downsampled to NPOS: (n, NPOS, 512)."""
    out = []
    for i in range(0, len(seqs), bs):
        x = batch_onehot(seqs[i:i + bs], device=m.device)
        with torch.no_grad():
            h = m.body_feats(x)                                  # (b,512,600)
            h = F.adaptive_avg_pool1d(h, NPOS)                   # (b,512,NPOS)
        out.append(h.transpose(1, 2).cpu())                      # (b,NPOS,512)
    return torch.cat(out)


def pooled(Fp):
    """mean+max over positions of the SAME per-position features -> (n,1024)."""
    return torch.cat([Fp.mean(1), Fp.amax(1)], dim=1)


def make_head(kind, dp, dev):
    if kind == "concat":
        return EarlyFusion(dr=1024, dp=dp).to(dev)
    if kind == "film":
        return ConditionedHead(dr=1024, dp=dp, residual=True).to(dev)
    return CrossAttnHead(d_model=512, dp=dp, heads=4, layers=2).to(dev)


def head_logit(kind, head, Xb, pe):
    if kind == "film":
        return head(Xb, pe)[0]
    return head(Xb, pe)


def train_eval(kind, Xtr, Ytr, Xte, te_y, ti_keep, E, K, seed, dev):
    """Xtr/Xte are the per-RBP-agnostic window features in the layout the head wants
    (pooled (n,1024) for concat/film, per-position (n,P,512) for xattn)."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    head = make_head(kind, E.shape[1], dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR[kind], weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
    pos = torch.nonzero(Ytr > 0, as_tuple=False).cpu().numpy()
    for ep in range(EPOCHS):
        rng.shuffle(pos)
        for i in range(0, len(pos), 256):
            pb = pos[i:i + 256]
            wp = torch.tensor(pb[:, 0], device=dev); kp = torch.tensor(pb[:, 1], device=dev)
            kn = torch.tensor(rng.integers(0, K, size=len(pb)), device=dev)
            Xb = torch.cat([Xtr[wp], Xtr[wp]]); pe = torch.cat([E[kp], E[kn]])
            yy = torch.cat([torch.ones(len(pb), device=dev), Ytr[wp, kn]])
            opt.zero_grad(); bce(head_logit(kind, head, Xb, pe), yy).backward(); opt.step()
    head.eval()

    def aps(Emap):
        out = np.full(K, np.nan)
        with torch.no_grad():
            for k in range(K):
                ss = []
                for i in range(0, len(Xte), 512):
                    Xb = Xte[i:i + 512]
                    ss.append(torch.sigmoid(head_logit(kind, head, Xb, Emap[k].expand(len(Xb), -1))).cpu().numpy())
                s = np.concatenate(ss); y = (te_y[:, ti_keep[k]] > 0).astype(float)
                if y.sum() >= 5:
                    out[k] = average_precision(s, y)
        return out
    perm = rng.permutation(K)
    return aps(E), aps(E[perm])


def summarize(REAL, SH, keep, ti_keep, te_y, kseeds):
    rmean = np.nanmean(REAL, 0); smean = np.nanmean(SH, 0)
    gap_seed = np.nanmean(REAL, 1) - np.nanmean(SH, 1)
    valid = ~np.isnan(rmean) & ~np.isnan(smean); diff = (rmean - smean)[valid]
    rng = np.random.default_rng(0)
    boot = np.array([np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    rows = [{"rbp": r, "cell": c, "real_mean": float(rmean[i]), "real_std": float(np.nanstd(REAL[:, i])),
             "shuffle_mean": float(smean[i]), "pos_rate": float((te_y[:, ti_keep[i]] > 0).mean())}
            for i, (ti, r, c) in enumerate(keep) if valid[i]]
    return {"real_mean": float(np.nanmean(rmean)), "shuffle_mean": float(np.nanmean(smean)),
            "gap_per_seed": gap_seed.tolist(), "gap_mean": float(gap_seed.mean()), "gap_std": float(gap_seed.std()),
            "gap_ci95": ci, "n_better": int((diff > 0).sum()), "n_rbp": int(valid.sum()), "rows": rows}


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "pureclip"
    kseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 25000
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 15000
    base = os.path.expanduser(f"~/ml4rg_data/binding/{scheme}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = read_rbp(f"{base}/rbp_cts.tsv"); pr = embeddings.ProteinRep()
    keep = [(ti, r, c) for ti, (r, c) in enumerate(tracks) if pr.esm(r) is not None]
    K = len(keep); ti_keep = [ti for ti, r, c in keep]
    m = load_parnet(device=dev)
    print(f"binding_mechanism [{scheme}] concat|film|xattn | K_rbp={K} seeds={kseeds} "
          f"train={n_train} test={n_test} dev={dev}", flush=True)
    tr_seq, tr_y = load_split(base, "train", n_train); te_seq, te_y = load_split(base, "test", n_test)
    print(f"computing per-position PARNET features (P={NPOS}) for {len(tr_seq)} train / {len(te_seq)} test ...", flush=True)
    Ftr = feats_pos(m, tr_seq); Fte = feats_pos(m, te_seq)
    Ptr = pooled(Ftr).to(dev); Pte = pooled(Fte).to(dev)        # (n,1024) for concat/film
    Ftr = Ftr.to(dev); Fte = Fte.to(dev)                        # (n,P,512) for xattn
    E = torch.stack([torch.tensor(pr.esm(r), dtype=torch.float32) for ti, r, c in keep]).to(dev)
    Ytr = torch.tensor((tr_y[:, ti_keep] > 0).astype(np.float32)).to(dev)

    results = {}
    for kind in ("concat", "film", "xattn"):
        Xtr, Xte = (Ftr, Fte) if kind == "xattn" else (Ptr, Pte)
        REAL = np.zeros((kseeds, K)); SH = np.zeros((kseeds, K))
        for s in range(kseeds):
            REAL[s], SH[s] = train_eval(kind, Xtr, Ytr, Xte, te_y, ti_keep, E, K, s, dev)
        res = summarize(REAL, SH, keep, ti_keep, te_y, kseeds)
        results[kind] = res
        print(f"  {kind:6}: real {res['real_mean']:.3f}  shuffle {res['shuffle_mean']:.3f}  "
              f"gap {res['gap_mean']:+.3f}+/-{res['gap_std']:.3f}  better {res['n_better']}/{res['n_rbp']}  "
              f"CI[{res['gap_ci95'][0]:+.3f},{res['gap_ci95'][1]:+.3f}]", flush=True)

    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    (o / "binding_mechanism.json").write_text(json.dumps(
        {"scheme": scheme, "seeds": kseeds, "n_train": len(tr_seq), "n_test": len(te_seq),
         "npos": NPOS, "lr": LR, "epochs": EPOCHS, "protein": "esm", "mechanisms": results}, indent=1))
    print("wrote binding_mechanism.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
