"""Supervised binding classifier on frozen PARNET body features (the M1/M2 classification step). The
zero-shot frozen-PARNET binding signal was weak (lift ~1.3x); here we TRAIN a multitask binary head on the
lab binary labels and measure per-RBP auPRC vs that baseline. Frozen body (mean+max pool over positions) ->
MLP -> 223 binding logits, BCE with per-task pos-weight (the ~1% imbalance), eval per-RBP auPRC on the test
split. Establishes whether PARNET's representation supports the canonical binding task once a head is trained.

  python -m mmpartnet.experiments.binding_head [scheme] [n_train] [n_test]
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
from mmpartnet.models.parnet import load_parnet
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.experiments.binding_eval import average_precision, read_rbp

LWIN = 600


def feats(m, seqs, bs=128):
    out = []
    for i in range(0, len(seqs), bs):
        x = batch_onehot(seqs[i:i + bs], device=m.device)
        with torch.no_grad():
            h = m.body_feats(x)                                  # (b, 512, L)
            out.append(torch.cat([h.mean(2), h.amax(2)], dim=1).cpu())   # (b, 1024) mean+max pool
    return torch.cat(out)


def load_split(base, split, n):
    ds = torch.load(f"{base}/dataset.pt", map_location="cpu", weights_only=False)[split]
    seqs, labs = [], []
    for el in ds:
        s = el["inputs"]["sequence"]
        if len(s) != LWIN:
            continue
        seqs.append(s); labs.append(torch.as_tensor(el["outputs"]["binding"]).numpy())
        if n and len(seqs) >= n:
            break
    return seqs, np.asarray(labs, np.float32)


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "pureclip"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    base = os.path.expanduser(f"~/ml4rg_data/binding/{scheme}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = read_rbp(f"{base}/rbp_cts.tsv"); T = len(tracks)
    m = load_parnet(device=dev)
    print(f"binding_head [{scheme}] T={T} n_train={n_train} n_test={n_test} dev={dev}", flush=True)

    tr_seq, tr_y = load_split(base, "train", n_train)
    te_seq, te_y = load_split(base, "test", n_test)
    print(f"built train={len(tr_seq)} test={len(te_seq)}; computing frozen-body features...", flush=True)
    Ftr = feats(m, tr_seq).to(dev); Fte = feats(m, te_seq).to(dev)
    Ytr = torch.tensor(tr_y).to(dev)
    pos = Ytr.sum(0); pw = ((len(Ytr) - pos) / (pos + 1e-6)).clamp(1, 200)   # per-task pos-weight

    head = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, T)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    head.train()
    for ep in range(15):
        perm = torch.randperm(len(Ftr), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad(); lossf(head(Ftr[b]), Ytr[b]).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pred = torch.sigmoid(head(Fte)).cpu().numpy()

    rows = []
    for ti, (r, c) in enumerate(tracks):
        y = (te_y[:, ti] > 0).astype(float)
        if y.sum() < 5:
            continue
        ap = average_precision(pred[:, ti], y); pr = float(y.mean())
        rows.append({"rbp": r, "cell": c, "pos_rate": pr, "auprc": ap, "lift": (ap / pr if pr > 0 else None)})
    apm = float(np.nanmean([x["auprc"] for x in rows]))
    liftm = float(np.nanmean([x["lift"] for x in rows if x["lift"]]))
    print(f"\n[{scheme}] TRAINED head: mean auPRC={apm:.3f}  mean LIFT={liftm:.1f}x  over {len(rows)} RBPs", flush=True)
    sv = sorted(rows, key=lambda r: -r["auprc"])
    print("  top8:   ", [(r["rbp"], round(r["auprc"], 3), round(r["lift"], 1)) for r in sv[:8]], flush=True)
    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    (o / f"binding_head_{scheme}.json").write_text(json.dumps(
        {"scheme": scheme, "n_train": len(tr_seq), "n_test": len(te_seq),
         "mean_auprc": apm, "mean_lift": liftm, "rows": rows}, indent=1))
    print(f"wrote binding_head_{scheme}.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
