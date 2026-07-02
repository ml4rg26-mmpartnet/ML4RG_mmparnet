"""Is the cross-attention head's position selection MECHANISTICALLY FAITHFUL, and is it PROTEIN-SPECIFIC?
The interpretability deliverable for the cross-attention head (the extra interpretability milestone).

The pooled-protein CrossAttnHead emits, per (window, protein), a (P,) attention map over RNA positions: which
positions the protein query selects. We test that map against the model's OWN attribution -- a full-pipeline
ISM (mutate each RNA position, re-run frozen PARNET body -> pool -> the SAME head logit, measure |drop|),
downsampled to the P attention positions. Two questions, two controls:

  faithful?        attention should agree with ISM (the positions ISM says carry the prediction).
  protein-specific? attention under the REAL protein should agree with ISM MORE than under a SHUFFLED protein
                    (else the attention is generic, not protein-conditioned).

Per the 'attention is not explanation' literature we report agreement as a SUPPLEMENTARY claim with TWO
metrics (Spearman over all P positions + Jaccard of the top-k positions) and always beside the perturbation
(ISM) ground truth and the protein-shuffle control -- never attention alone.

  python -m mmpartnet.experiments.xattn_faithfulness [scheme] [n_rbp] [win_per_rbp]
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
from mmpartnet.models.cross_attn_head import CrossAttnHead
from mmpartnet.models.parnet import load_parnet
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.experiments.binding_eval import average_precision, read_rbp
from mmpartnet.experiments.binding_head import load_split

LWIN = 600
NPOS = 64


def feats_seq(m, seqs, bs=128):
    out = []
    for i in range(0, len(seqs), bs):
        x = batch_onehot(seqs[i:i + bs], device=m.device)
        with torch.no_grad():
            h = F.adaptive_avg_pool1d(m.body_feats(x), NPOS)
        out.append(h.transpose(1, 2).cpu())
    return torch.cat(out)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def jaccard_topk(a, b, k):
    ta = set(np.argsort(a)[-k:].tolist()); tb = set(np.argsort(b)[-k:].tolist())
    return len(ta & tb) / len(ta | tb)


def head_logit_seqs(m, head, seqs, e, dev, bs=256):
    """Full-pipeline logit for a batch of sequences under protein e: PARNET body -> pool(P) -> head."""
    outs = []
    for i in range(0, len(seqs), bs):
        x = batch_onehot(seqs[i:i + bs], device=dev)
        with torch.no_grad():
            H = F.adaptive_avg_pool1d(m.body_feats(x), NPOS).transpose(1, 2)
            outs.append(head(H, e.expand(len(x), -1)).cpu().numpy())
    return np.concatenate(outs)


def pipeline_ism(m, head, seq, e, dev):
    """Full multimodal-pipeline ISM over the 600 RNA positions for THIS head + protein. Returns (600,)
    mean |drop in head logit| over the 3 alternative bases per position, then we downsample to NPOS."""
    bases = "ACGT"
    ref = head_logit_seqs(m, head, [seq], e, dev)[0]
    variants, pos = [], []
    for i, ch in enumerate(seq):
        for b in bases:
            if b != ch:
                variants.append(seq[:i] + b + seq[i + 1:]); pos.append(i)
    sv = head_logit_seqs(m, head, variants, e, dev)
    imp = np.zeros(len(seq))
    for j, i in enumerate(pos):
        imp[i] += abs(ref - sv[j])
    imp = imp / 3.0
    # block-mean to NPOS
    return imp.reshape(NPOS, len(seq) // NPOS).mean(1) if len(seq) % NPOS == 0 else \
        F.adaptive_avg_pool1d(torch.tensor(imp)[None, None], NPOS)[0, 0].numpy()


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "pureclip"
    n_rbp = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    win_per = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    base = os.path.expanduser(f"~/ml4rg_data/binding/{scheme}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = read_rbp(f"{base}/rbp_cts.tsv"); pr = embeddings.ProteinRep()
    keep = [(ti, r, c) for ti, (r, c) in enumerate(tracks) if pr.esm(r) is not None]
    K = len(keep); ti_keep = [ti for ti, r, c in keep]
    m = load_parnet(device=dev)
    print(f"xattn_faithfulness [{scheme}] K_rbp={K} dev={dev}; training cross-attn head ...", flush=True)

    tr_seq, tr_y = load_split(base, "train", 25000); te_seq, te_y = load_split(base, "test", 15000)
    Ftr = feats_seq(m, tr_seq).to(dev)
    E = torch.stack([torch.tensor(pr.esm(r), dtype=torch.float32) for ti, r, c in keep]).to(dev)
    Ytr = torch.tensor((tr_y[:, ti_keep] > 0).astype(np.float32)).to(dev)

    head = CrossAttnHead(d_model=512, dp=E.shape[1], heads=4, layers=2).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=5e-4, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0); torch.manual_seed(0)
    pos = torch.nonzero(Ytr > 0, as_tuple=False).cpu().numpy()
    for ep in range(10):
        rng.shuffle(pos)
        for i in range(0, len(pos), 128):
            pb = pos[i:i + 128]
            wp = torch.tensor(pb[:, 0], device=dev); kp = torch.tensor(pb[:, 1], device=dev)
            kn = torch.tensor(rng.integers(0, K, size=len(pb)), device=dev)
            Hb = torch.cat([Ftr[wp], Ftr[wp]]); pe = torch.cat([E[kp], E[kn]])
            yy = torch.cat([torch.ones(len(pb), device=dev), Ytr[wp, kn]])
            opt.zero_grad(); bce(head(Hb, pe), yy).backward(); opt.step()
    head.eval()

    # pick the n_rbp RBPs with the most test positives (most reliable motif)
    posc = [(int((te_y[:, ti] > 0).sum()), i, r) for i, (ti, r, c) in enumerate(keep)]
    posc.sort(reverse=True)
    chosen = [(i, r) for _, i, r in posc[:n_rbp]]
    print(f"chosen RBPs: {[r for _, r in chosen]}", flush=True)

    rows = []; example = None
    for ki, rbp in chosen:
        ti = ti_keep[ki]
        widx = np.nonzero(te_y[:, ti] > 0)[0][:win_per]
        e_real = E[ki]; ko = (ki + K // 2) % K; e_shuf = E[ko]      # a different protein as the control
        for wi in widx:
            seq = te_seq[int(wi)]
            if len(seq) != LWIN:
                continue
            H = feats_seq(m, [seq]).to(dev)
            with torch.no_grad():
                _, a_real = head(H, e_real.expand(1, -1), return_attn=True)
                _, a_shuf = head(H, e_shuf.expand(1, -1), return_attn=True)
            a_real = a_real[0].cpu().numpy(); a_shuf = a_shuf[0].cpu().numpy()
            ism = pipeline_ism(m, head, seq, e_real, dev)
            k = max(3, NPOS // 8)
            rows.append({
                "rbp": rbp,
                "sp_real": spearman(a_real, ism), "sp_shuf": spearman(a_shuf, ism),
                "jac_real": jaccard_topk(a_real, ism, k), "jac_shuf": jaccard_topk(a_shuf, ism, k),
                "attn_conc_real": float(np.sort(a_real)[-k:].sum()), "attn_conc_shuf": float(np.sort(a_shuf)[-k:].sum()),
            })
            if example is None:
                example = {"rbp": rbp, "attn_real": a_real.tolist(), "attn_shuf": a_shuf.tolist(), "ism": ism.tolist()}
        print(f"  {rbp}: n_win={len(widx)} done", flush=True)

    def agg(key):
        v = np.array([r[key] for r in rows], float); return float(np.nanmean(v)), float(np.nanstd(v))
    sp_r, sp_rs = agg("sp_real"); sp_s, sp_ss = agg("sp_shuf")
    jr, jrs = agg("jac_real"); js, jss = agg("jac_shuf")
    print(f"\n[faithfulness] Spearman(attn,ISM): real {sp_r:+.3f}+/-{sp_rs:.3f}  shuffle {sp_s:+.3f}+/-{sp_ss:.3f}", flush=True)
    print(f"[faithfulness] Jaccard top-k(attn,ISM): real {jr:.3f}  shuffle {js:.3f}", flush=True)
    print(f"  => real>shuffle on Spearman in {sum(r['sp_real']>r['sp_shuf'] for r in rows)}/{len(rows)} windows", flush=True)

    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    (o / "xattn_faithfulness.json").write_text(json.dumps(
        {"scheme": scheme, "n_rbp": n_rbp, "win_per": win_per, "npos": NPOS, "n_windows": len(rows),
         "spearman_real": [sp_r, sp_rs], "spearman_shuf": [sp_s, sp_ss],
         "jaccard_real": [jr, jrs], "jaccard_shuf": [js, jss],
         "n_real_gt_shuf": int(sum(r["sp_real"] > r["sp_shuf"] for r in rows)),
         "rows": rows, "example": example}, indent=1))
    print("wrote xattn_faithfulness.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
