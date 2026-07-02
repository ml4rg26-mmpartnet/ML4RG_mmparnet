"""Frozen-PARNET binding CLASSIFICATION on the lab binary labels (narrowpeak_intersect / pureclip).
Zero training: per-window binding score = max raw target logit per RBP (binding magnitude proxy); per-RBP
auPRC vs the binary label over the test split. Labels are very imbalanced (sub-1% positive), so we report
auPRC + LIFT over the positive-rate baseline (= random auPRC). Tests whether PARNET's internal signal
predicts the canonical binding task out of the box, and gives the per-RBP auPRC M1's sparsity goal needs.

  python -m mmpartnet.experiments.binding_eval [scheme] [max_n]   (scheme: pureclip | narrowpeak)
"""
from __future__ import annotations
import sys, os, json
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from mmpartnet import config
from mmpartnet.models.parnet import load_parnet
from mmpartnet.process.onehot import batch_onehot


def average_precision(scores, y):
    """Area under the precision-recall curve (sklearn-free). y binary."""
    if y.sum() < 1:
        return float("nan")
    order = np.argsort(-scores)
    yo = y[order]
    tp = np.cumsum(yo); fp = np.cumsum(1 - yo)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / y.sum()
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def read_rbp(path):
    rows = []
    with open(path) as f:
        h = f.readline().rstrip("\n").split("\t"); ci = {c: i for i, c in enumerate(h)}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) > ci["rbp"]:
                rows.append((p[ci["rbp"]], p[ci["ct"]]))
    return rows


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "pureclip"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    base = os.path.expanduser(f"~/ml4rg_data/binding/{scheme}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = read_rbp(f"{base}/rbp_cts.tsv"); T = len(tracks)
    m = load_parnet(device=dev)
    sel = np.array([m.idx.get(f"{r}_{c}", -1) for r, c in tracks])
    valid = sel >= 0; selA = np.where(valid, sel, 0)
    ds = torch.load(f"{base}/dataset.pt", map_location="cpu", weights_only=False)
    test = ds["test"]; N = len(test) if not max_n else min(max_n, len(test))
    print(f"binding_eval [{scheme}] tracks={T} mapped={int(valid.sum())} N={N} dev={dev}", flush=True)

    LWIN = 600
    sc_l, lb_l = [], []                     # collected only for len==LWIN windows (some tiles are shorter)
    B = 128; buf, buflab, skipped = [], [], 0

    def flush():
        x = batch_onehot(buf, device=m.device)
        with torch.no_grad():
            tg = m.run_raw(x)["target"].amax(dim=2).detach().cpu().numpy()   # (b, 223) max logit/track
        for j in range(len(buf)):
            sc_l.append(tg[j, selA]); lb_l.append(buflab[j])
        buf.clear(); buflab.clear()

    for i in range(N):
        el = test[i]
        seq = el["inputs"]["sequence"]
        if len(seq) != LWIN:
            skipped += 1; continue
        lab = el["outputs"]["binding"]
        lab = (lab.to_dense() if torch.is_tensor(lab) and lab.is_sparse else torch.as_tensor(lab)).numpy()
        buf.append(seq); buflab.append(lab)
        if len(buf) >= B:
            flush()
        if i % 10000 == 0:
            print(f"  {i}/{N}", flush=True)
    if buf:
        flush()
    scores = np.asarray(sc_l, np.float32); labels = np.asarray(lb_l, np.float32)
    print(f"  scored {len(scores)} windows (skipped {skipped} non-{LWIN}nt)", flush=True)

    rows = []
    for ti, (r, c) in enumerate(tracks):
        if not valid[ti]:
            continue
        y = (labels[:, ti] > 0).astype(float); pr = float(y.mean())
        if y.sum() < 5:
            continue
        ap = average_precision(scores[:, ti], y)
        rows.append({"rbp": r, "cell": c, "pos_rate": pr, "auprc": ap,
                     "lift": (ap / pr if pr > 0 else None)})
    apm = float(np.nanmean([x["auprc"] for x in rows]))
    prm = float(np.mean([x["pos_rate"] for x in rows]))
    liftm = float(np.nanmean([x["lift"] for x in rows if x["lift"]]))
    print(f"\n[{scheme}] mean auPRC={apm:.3f}  mean pos-rate(baseline)={prm:.4f}  mean LIFT={liftm:.1f}x  "
          f"over {len(rows)} RBPs", flush=True)
    sv = sorted(rows, key=lambda r: -r["auprc"])
    print("  top8:   ", [(r["rbp"], round(r["auprc"], 3), round(r["lift"], 1)) for r in sv[:8]], flush=True)
    print("  bottom5:", [(r["rbp"], round(r["auprc"], 3)) for r in sv[-5:]], flush=True)
    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    (o / f"binding_eval_{scheme}.json").write_text(json.dumps(
        {"scheme": scheme, "n": N, "mean_auprc": apm, "mean_pos_rate": prm, "mean_lift": liftm,
         "rows": rows}, indent=1))
    print(f"wrote binding_eval_{scheme}.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
