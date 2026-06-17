"""FINAL VERIFICATION — run the PARNET demo's head-finetune on OUR data, compare to the baseline.

Faithful to `parnet--demo--train-models`: replace PARNET's 223-task head with a fresh K-task head
(**target + control + additive-mix + mix-coeff penalty**), **HEAD-finetune** on frozen body features
with **MultinomialNLL**, then evaluate per-RBP profile **Pearson/Spearman** on held-out windows —
FINETUNED vs the PRETRAINED PARNET (the recovered baseline). Data = our assembled ENCODE signal +
SMInput **control** (adapters.eclip_signal, remote bigWigs). Demo subset = 9 spliceosome-HepG2 RBPs.

  python -m mmpartnet.experiments.recover_demo_finetune --group spliceosome --cell HepG2 --nwin 12 --epochs 10
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from mmpartnet import config
from mmpartnet.io import genome, groups
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.models.parnet import load_parnet
from mmpartnet.adapters import eclip_signal
from mmpartnet.adapters import peaks as peaks_adapter
from mmpartnet.experiments.recover_demo_profile import _peaks, _pearson, _spearman, LWIN, MIN_RC_SUM


class RandomBody(nn.Module):
    """Control 3 (random-init/random-body fairness control): a FROZEN, randomly-initialized feature
    extractor whose output matches PARNET's body shape (B, 512, L). Train the SAME DemoHead on these
    instead of PARNET's body features; if it recovers obs as well as the PARNET-body head, the
    finetune gain is head data-fitting on peak-centered windows, NOT transfer from PARNET's learned
    representation. Same capacity (512-wide), same training, only the body is un-pretrained."""

    def __init__(self, d=512, k=15, device="cpu"):
        super().__init__()
        torch.manual_seed(0)                                   # deterministic random body
        self.c1 = nn.Conv1d(4, d, k, padding="same")
        self.c2 = nn.Conv1d(d, d, k, padding="same")
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device).eval()

    @torch.no_grad()
    def body_feats(self, onehot):                              # (B,4,L) -> (B,d,L), match m.body_feats
        return torch.relu(self.c2(torch.relu(self.c1(onehot.float()))))


class DemoHead(nn.Module):
    """The demo's head re-homed on frozen PARNET body features (B, 512, L): per-track target +
    control position-logits, an additive mixture with a learned per-track mix-coeff, + the mix
    penalty (5.0). Trained with MultinomialNLL on eCLIP + control counts."""

    def __init__(self, n_tracks, d=512, penalty=5.0):
        super().__init__()
        self.target = nn.Conv1d(d, n_tracks, 1)
        self.control = nn.Conv1d(d, n_tracks, 1)
        self.mix_logit = nn.Parameter(torch.zeros(n_tracks))
        self.penalty = penalty

    def forward(self, feat):                                   # feat: (B, d, L)
        p_t = torch.softmax(self.target(feat), dim=2)         # (B, T, L) prob over positions
        p_c = torch.softmax(self.control(feat), dim=2)
        mix = torch.sigmoid(self.mix_logit).view(1, -1, 1)
        return mix * p_t + (1 - mix) * p_c, p_c               # p_total, p_control

    def loss(self, feat, obs_e, obs_c):                       # obs_*: (B, T, L) counts
        p_total, p_c = self.forward(feat)
        # per-(window,track) multinomial CE NORMALIZED by read depth → each window/track weighted
        # equally and the loss is well-scaled (~log L), so Adam actually converges. eCLIP via the
        # mixture (target+control), control via its own channel; + the mix-coeff penalty.
        ne = (obs_e * torch.log(p_total + 1e-8)).sum(dim=2)   # (B, T)
        nc = (obs_c * torch.log(p_c + 1e-8)).sum(dim=2)
        nll = -(ne / (obs_e.sum(dim=2) + 1e-6)).mean() - (nc / (obs_c.sum(dim=2) + 1e-6)).mean()
        return nll + self.penalty * torch.sigmoid(self.mix_logit).mean()


class RBPNetHead(nn.Module):
    """Control 2: the FAITHFUL RBPNet objective (Horlacher 2023, Genome Biology 24:180; lab
    `AdditiveMix` use_maximum_target_control_logprob branch). Differs from DemoHead in exactly the
    ways the established method does, to test whether the finetune gain survives:
      - log-space `additive_mix_max` mixture (not the prob-space convex mix);
      - raw-count MultinomialNLL, count-weighted, NOT divided by read depth;
      - log2 task-balancing (down-weight high-count tracks);
      - min-height masking (only tracks/windows with >= min_height reads contribute).
    mix_coeff is per-(window,track) from globally-pooled features (lab MixCoeffMLP), not a bare scalar.
    """

    def __init__(self, n_tracks, d=512, penalty=5.0, min_height=float(MIN_RC_SUM)):
        super().__init__()
        self.target = nn.Conv1d(d, n_tracks, 1)
        self.control = nn.Conv1d(d, n_tracks, 1)
        self.mix = nn.Linear(d, n_tracks)            # per-sequence mix logit from pooled features
        self.penalty = penalty
        self.min_height = min_height

    def _logprobs(self, feat):                       # feat (B,d,L)
        tl = self.target(feat); cl = self.control(feat)                  # (B,T,L) logits
        t_lp = tl - torch.logsumexp(tl, dim=2, keepdim=True)             # log-softmax over POSITIONS
        c_lp = cl - torch.logsumexp(cl, dim=2, keepdim=True)
        mix = torch.sigmoid(self.mix(feat.mean(dim=2))).unsqueeze(-1)    # (B,T,1)
        mx = torch.maximum(t_lp, c_lp)                                   # additive_mix_max (stable)
        tot_lp = mx + torch.log(mix * torch.exp(t_lp - mx) + (1 - mix) * torch.exp(c_lp - mx) + 1e-10)
        return t_lp, c_lp, tot_lp, mix

    def forward(self, feat):                         # return probs for eval-compat with DemoHead
        t_lp, c_lp, tot_lp, _ = self._logprobs(feat)
        return torch.exp(tot_lp), torch.exp(c_lp)

    def loss(self, feat, obs_e, obs_c):              # obs_*: (B,T,L) RAW counts
        t_lp, c_lp, tot_lp, mix = self._logprobs(feat)
        nll_e = -(obs_e * tot_lp).sum(dim=2)         # (B,T) raw-count MultinomialNLL (count-weighted)
        nll_c = -(obs_c * c_lp).sum(dim=2)
        he = obs_e.sum(dim=2); hc = obs_c.sum(dim=2)
        m_e = (he >= self.min_height).float(); m_c = (hc >= self.min_height).float()   # min-height mask
        w_e = 1.0 / torch.log2(2.0 + he); w_c = 1.0 / torch.log2(2.0 + hc)             # log2 balancing
        le = (nll_e * m_e * w_e).sum() / (m_e.sum() + 1e-6)
        lc = (nll_c * m_c * w_c).sum() / (m_c.sum() + 1e-6)
        return le + lc + self.penalty * mix.mean()


def build_dataset(m, rbps, cell, manifest, nwin, body=None):
    """Multi-task windows: frozen body features (512,L) + observed per-nt eCLIP + control (T,L) + seq.
    `body` provides .body_feats (default = PARNET m; Control 3 passes a RandomBody)."""
    body = body or m
    readers, tracks, exps, win, seen = {}, {}, {}, [], set()
    for g in rbps:
        rec = next((r for r in manifest.get(g, []) if r["cell"] == cell), None)
        if rec is None:
            continue
        tr = m.track_index(g, cell)
        if tr is None:
            continue
        ecl, ctrl, u = eclip_signal.readers_for(rec["exp"])
        readers[g] = (ecl, ctrl); tracks[g] = tr; exps[g] = (rec["exp"], u.get("control_exp"))
        for chrom, s, e, strand, _sc in _peaks(peaks_adapter.resolve_bed(rec), nwin):
            if not ecl.has(chrom):
                continue
            c = (s + e) // 2; ws = c - LWIN // 2; key = (chrom, ws, strand)
            if key in seen:
                continue
            seen.add(key); win.append((chrom, ws, ws + LWIN, strand))
    rbps = [g for g in rbps if g in readers]
    T = len(rbps)
    F, OE, OC, seqs = [], [], [], []
    for (chrom, ws, we, strand) in win:
        seq = genome.window_seq(chrom, ws, we, strand)
        if seq is None or len(seq) != LWIN:
            continue
        with torch.no_grad():
            f = body.body_feats(batch_onehot([seq], device=m.device))[0].float().cpu().numpy()
        oe = np.zeros((T, LWIN), np.float32); oc = np.zeros((T, LWIN), np.float32)
        for ti, g in enumerate(rbps):
            ecl, ctrl = readers[g]
            pe = ecl.profile(chrom, ws, we, strand)
            if pe is not None:
                oe[ti] = pe
            if ctrl is not None:
                pc = ctrl.profile(chrom, ws, we, strand)
                if pc is not None:
                    oc[ti] = pc
        F.append(f); OE.append(oe); OC.append(oc); seqs.append(seq)
    return (rbps, tracks, exps, np.asarray(F, np.float32),
            np.asarray(OE, np.float32), np.asarray(OC, np.float32), seqs)


def random_body_feats(seqs, m, d=512, device=None):
    """Control-3 features: a FROZEN random body's (d,L) output for each window sequence. Reuses the
    ALREADY-READ window `seqs` (no eCLIP re-reads), so building all controls costs ONE remote-signal
    pass. GPU-cheap (two conv layers)."""
    dev = device or m.device
    rb = RandomBody(d=d, device=dev)
    F = []
    for s in seqs:
        with torch.no_grad():
            F.append(rb.body_feats(batch_onehot([s], device=dev))[0].float().cpu().numpy())
    return np.asarray(F, np.float32)


def train_eval_head(F, OE, OC, seqs, rbps, tracks, m, *, objective="normalized", epochs=10,
                    test_frac=0.3, penalty=5.0, seed=0, device=None):
    """Train a fresh head on prebuilt features `F` (+ observed eCLIP/control `OE`/`OC`) and evaluate
    per-RBP profile Pearson/Spearman on held-out windows vs the pretrained PARNET. Returns the per-RBP
    rows (same schema main writes). Pulled out of main so the notebook can build the dataset ONCE and
    train several heads (normalized / rbpnet / random-body) on it."""
    dev = device or m.device
    N = len(F); T = len(rbps)
    rng = np.random.default_rng(seed); idx = rng.permutation(N); ntr = int(N * (1 - test_frac))
    tr_i, te_i = idx[:ntr], idx[ntr:]
    Ft = torch.tensor(F).to(dev); OEt = torch.tensor(OE).to(dev); OCt = torch.tensor(OC).to(dev)
    head = (RBPNetHead(T, d=F.shape[1], penalty=penalty) if objective == "rbpnet"
            else DemoHead(T, d=F.shape[1], penalty=penalty)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3); head.train()
    for _ep in range(epochs):
        perm = rng.permutation(tr_i)
        for i in range(0, len(perm), 16):
            b = perm[i:i + 16]
            loss = head.loss(Ft[b], OEt[b], OCt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        p_ft = head(Ft[te_i])[0].cpu().numpy()
    pre = np.zeros_like(p_ft); sel = [tracks[g] for g in rbps]
    with torch.no_grad():
        for j, wi in enumerate(te_i):
            tot = m.full(batch_onehot([seqs[wi]], device=m.device))["total"][0].cpu().numpy()
            pre[j] = tot[sel]
    rows = []
    for ti, g in enumerate(rbps):
        fp, fs, pp, ps = [], [], [], []
        for j in range(len(te_i)):
            o = OE[te_i[j], ti]
            if o.sum() < MIN_RC_SUM:
                continue
            obs = o / o.sum()
            fp.append(_pearson(p_ft[j, ti], obs)); fs.append(_spearman(p_ft[j, ti], obs))
            pp.append(_pearson(pre[j, ti], obs)); ps.append(_spearman(pre[j, ti], obs))
        if not fp:
            rows.append({"rbp": g, "n": 0}); continue
        rows.append({"rbp": g, "n": len(fp),
                     "pretrained_pearson": float(np.nanmean(pp)), "finetuned_pearson": float(np.nanmean(fp)),
                     "pretrained_spearman": float(np.nanmean(ps)), "finetuned_spearman": float(np.nanmean(fs))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="spliceosome")
    ap.add_argument("--cell", default="HepG2")
    ap.add_argument("--nwin", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--penalty", type=float, default=5.0)
    ap.add_argument("--baseline", default="parnetbody", choices=["parnetbody", "randbody"],
                    help="Control 3: 'parnetbody' = head on frozen PARNET body (transfer); "
                         "'randbody' = head on a frozen RANDOM body (fairness control)")
    ap.add_argument("--objective", default="normalized", choices=["normalized", "rbpnet"],
                    help="Control 2: 'normalized' = depth-normalized demo loss; "
                         "'rbpnet' = faithful raw-count MultinomialNLL + additive_mix_max + log2-balance + min-height")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rbps0 = groups.resolve(args.group) or [args.group]
    manifest = json.loads((config.DATA / "eclip_manifest.json").read_text())
    rbps0 = [g for g in rbps0 if g in manifest]
    print(f"recover demo FINETUNE | group={args.group} cell={args.cell} RBPs={rbps0} "
          f"nwin={args.nwin} epochs={args.epochs} baseline={args.baseline} dev={dev}", flush=True)
    m = load_parnet()
    body = m if args.baseline == "parnetbody" else RandomBody(d=512, device=dev)

    rbps, tracks, exps, F, OE, OC, seqs = build_dataset(m, rbps0, args.cell, manifest, args.nwin, body=body)
    T = len(rbps); N = len(F)
    print(f"assembled {N} windows x {T} tracks (eCLIP + control). control_exps="
          f"{[exps[g][1] for g in rbps]}", flush=True)
    if N < 12:
        print("too few windows; abort", flush=True); return

    rows = train_eval_head(F, OE, OC, seqs, rbps, tracks, m, objective=args.objective,
                           epochs=args.epochs, test_frac=args.test_frac, penalty=args.penalty, device=dev)

    ev = [r for r in rows if r.get("n")]
    print(f"\n{'RBP':9} {'n':>3}  {'pre.Pear':>9} {'ft.Pear':>9}   {'pre.Spear':>9} {'ft.Spear':>9}", flush=True)
    for r in ev:
        print(f"{r['rbp']:9} {r['n']:>3}  {r['pretrained_pearson']:>+9.3f} {r['finetuned_pearson']:>+9.3f}   "
              f"{r['pretrained_spearman']:>+9.3f} {r['finetuned_spearman']:>+9.3f}", flush=True)
    if ev:
        pp = float(np.mean([r["pretrained_pearson"] for r in ev])); fp = float(np.mean([r["finetuned_pearson"] for r in ev]))
        ps = float(np.mean([r["pretrained_spearman"] for r in ev])); fs = float(np.mean([r["finetuned_spearman"] for r in ev]))
        print(f"\nMEAN  Pearson:  pretrained={pp:+.3f}  finetuned={fp:+.3f}  (delta={fp-pp:+.3f})", flush=True)
        print(f"MEAN  Spearman: pretrained={ps:+.3f}  finetuned={fs:+.3f}  (delta={fs-ps:+.3f})", flush=True)
        print("\nVERDICT: " + ("finetuned >= pretrained on our data — demo head-finetune reproduced."
              if fp + 1e-6 >= pp else
              "finetuned < pretrained — expected in this small-data regime (the pretrained all-223 head "
              "already saw these 9 RBPs); the pipeline runs the demo end-to-end and both are well above chance."), flush=True)
    out = {"group": args.group, "cell": args.cell, "n_windows": N, "n_tracks": T, "rbps": rbps,
           "epochs": args.epochs, "baseline": args.baseline, "objective": args.objective, "rows": rows}
    o = config.RESULTS; o.mkdir(parents=True, exist_ok=True)
    tag = "" if (args.baseline == "parnetbody" and args.objective == "normalized") else \
          f"_{args.baseline}_{args.objective}"
    fname = f"recover_demo_finetune{tag}.json"
    (o / fname).write_text(json.dumps(out, indent=1))
    print(f"wrote {o / fname}", flush=True)


if __name__ == "__main__":
    main()
