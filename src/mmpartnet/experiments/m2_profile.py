"""M2 (the real multimodal contribution) — nt-resolution PROFILE conditioned on protein, on the lab
full-223 encode.filtered.hfds. A SINGLE protein-conditioned head predicts a per-nt eCLIP profile for ANY
RBP from its protein rep (vs PARNET's 223 fixed per-RBP heads). The multimodal question at profile
resolution: does conditioning on the RIGHT protein give a better per-nt profile than a WRONG (shuffled)
protein? -- with the same derangement + within-family controls as the binding grand-check, and the
pretrained-PARNET per-track profile as the reference ceiling.

Architectures that can EMIT a per-position profile (pooled-query xattn cannot):
  film    : FiLM-modulate frozen body feats by the protein rep -> Conv1d -> per-nt target logits
  perres  : RNA positions attend the protein's (L,32) residue tokens (BiCrossAttn, per-position stream
            kept) -> per-nt target logits  (TFBindFormer/CORAL-faithful, profile variant)
Control channel (background) is protein-agnostic; additive mixture + MultinomialNLL (the RBPNet objective).

  python -m mmpartnet.experiments.m2_profile [cell] [n_windows] [arch] [epochs]
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
from mmpartnet.io import embeddings, cohort
from mmpartnet.models.parnet import load_parnet
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.data.sources import hfds

LWIN = 600
LMAX = 384
MIN_RC = 8.0          # min profile read-count for a (window,rbp) pair to count
HELD_CHROM = {"chr2", "chr9", "chr16"}


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


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


class FiLMProfile(nn.Module):
    """Protein FiLM-modulates frozen body feats -> per-nt target logits; protein-agnostic control."""
    def __init__(self, d=512, dp=1280):
        super().__init__()
        self.film = nn.Linear(dp, 2 * d)
        self.tgt = nn.Conv1d(d, d, 5, padding="same"); self.tgt2 = nn.Conv1d(d, 1, 1)
        self.ctrl = nn.Conv1d(d, 1, 1)
        self.mix = nn.Linear(dp, 1)

    def forward(self, H, e):                     # H:(B,d,L)  e:(B,dp)
        g, b = self.film(e).chunk(2, -1)
        h = g.unsqueeze(-1) * H + b.unsqueeze(-1)
        t = self.tgt2(torch.relu(self.tgt(h))).squeeze(1)        # (B,L) target logit
        c = self.ctrl(H).squeeze(1)                              # (B,L) control logit
        mix = torch.sigmoid(self.mix(e))                         # (B,1)
        return t, c, mix


class PerresProfile(nn.Module):
    """RNA positions attend the protein's residue tokens -> per-position target logits (profile variant of
    BiCrossAttnHead; the per-position stream is kept, no [BIND] pool)."""
    def __init__(self, d=512, dp=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.rp = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.pp = nn.Sequential(nn.Linear(dp, d), nn.LayerNorm(d))
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.tgt = nn.Linear(d, 1)
        self.ctrl = nn.Conv1d(d, 1, 1)
        self.mix = nn.Parameter(torch.zeros(1))

    def forward(self, H, P, mask=None):           # H:(B,d,L)  P:(B,Lp,dp)
        x = self.rp(H.transpose(1, 2))            # (B,L,d) queries
        kv = self.pp(P)
        for a, n1, ff, n2 in zip(self.attn, self.n1, self.ff, self.n2):
            ctx, _ = a(x, kv, kv, key_padding_mask=mask, need_weights=False)
            x = n1(x + ctx); x = n2(x + ff(x))
        t = self.tgt(x).squeeze(-1)               # (B,L)
        c = self.ctrl(H).squeeze(1)               # (B,L)
        return t, c, torch.sigmoid(self.mix).expand(H.size(0), 1)


def nll_profile(t, c, mix, obs_e, obs_c):
    """additive-mixture MultinomialNLL (count-weighted, normalized) over positions."""
    p_t = torch.log_softmax(t, dim=1); p_c = torch.log_softmax(c, dim=1)
    mx = torch.maximum(p_t, p_c)
    tot = mx + torch.log(mix * torch.exp(p_t - mx) + (1 - mix) * torch.exp(p_c - mx) + 1e-10)
    he = obs_e.sum(1, keepdim=True) + 1e-6; hc = obs_c.sum(1, keepdim=True) + 1e-6
    le = -(obs_e * tot).sum(1, keepdim=True) / he
    lc = -(obs_c * p_c).sum(1, keepdim=True) / hc
    return (le + lc).mean() + 0.1 * mix.mean()


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "HepG2"
    n_win = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    arch = sys.argv[3] if len(sys.argv) > 3 else "both"
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    bundle = os.environ.get("ML4RG_DATA", os.path.expanduser("~/ml4rg_data"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import datasets
    from mmpartnet.data.sources.hfds import read_full_rbp_cts, _cell_select, _to_dense
    pr = embeddings.ProteinRep()
    z = np.load(f"{bundle}/perres32.npz", allow_pickle=True); have = set(z.files)
    cts = read_full_rbp_cts()
    syms0 = [r["rbp"] for r in cts if r["ct"] == cell and pr.esm(r["rbp"]) is not None and r["rbp"] in have]
    seen = set(); syms0 = [s for s in syms0 if not (s in seen or seen.add(s))]
    local, syms_w, _sy, _fu = _cell_select(syms0, cts, cell)     # local = dense track indices for our syms
    syms = syms_w; K = len(syms)
    fam_map = cohort.attract_families(syms)
    fam_lab = [(fam_map[s][0] if s in fam_map and fam_map[s] else "other") for s in syms]
    fam_ids = np.array([sorted(set(fam_lab)).index(f) for f in fam_lab])
    m = load_parnet(device=dev)
    print(f"m2_profile cell={cell} K_rbp={K} n_win={n_win} arch={arch} epochs={epochs} dev={dev}", flush=True)

    # load the hfds Arrow dir directly (flat Dataset; avoids the repo iter_elements split-bug). STRIDE-sample
    # across the whole set so all chromosomes are represented (the rows are position-ordered).
    ds = datasets.load_from_disk(os.environ.get("ML4RG_LAB_HFDS", str(config.LAB_HFDS)))
    total = len(ds); stride = max(1, total // n_win)
    seqs, OE, OC, CHR = [], [], [], []
    for ri, el in enumerate(ds):                                 # sequential mmap stream (fast); subsample by stride
        if ri % stride != 0:
            continue
        seq = el["inputs"]["sequence"]
        if len(seq) != LWIN:
            continue
        seqs.append(seq)
        OE.append(_to_dense(el["outputs"]["eCLIP"])[local].astype(np.float32))
        OC.append(_to_dense(el["outputs"]["control"])[local].astype(np.float32))
        CHR.append(str(el.get("meta", {}).get("name", "")).split(":")[0])
        if len(seqs) >= n_win:
            break
    OE = np.asarray(OE); OC = np.asarray(OC); CHR = np.asarray(CHR)
    # batched body features
    Fb = []
    for i in range(0, len(seqs), 128):
        with torch.no_grad():
            Fb.append(m.body_feats(batch_onehot(seqs[i:i + 128], device=dev)).float().cpu().numpy())
    Fb = np.concatenate(Fb)
    M = len(Fb)
    print(f"streamed {M} windows; building pairs (min_rc={MIN_RC}) ...", flush=True)

    # (window, rbp) pairs where the rbp has signal; held-out-WINDOW split (chrom-holdout if the held chroms
    # are well represented, else a random 30% window split -- both are leakage-safe held-out windows).
    sums = OE.sum(2)                                  # (M,K)
    pairs = np.argwhere(sums >= MIN_RC)               # [[wi,ki],...]
    is_test = np.array([CHR[wi] in HELD_CHROM for wi, ki in pairs])
    split_kind = "chrom-holdout"
    if is_test.sum() < 200:
        rng0 = np.random.default_rng(0)
        test_win = set(rng0.choice(M, size=max(1, int(0.3 * M)), replace=False).tolist())
        is_test = np.array([wi in test_win for wi, ki in pairs])
        split_kind = "random-window-30pct"
    tr = pairs[~is_test]; te = pairs[is_test]
    print(f"split={split_kind} pairs train={len(tr)} test={len(te)} over {K} RBPs "
          f"(chroms seen: {sorted(set(CHR))[:6]}...)", flush=True)

    Fb_t = torch.tensor(Fb).to(dev)
    E = torch.stack([torch.tensor(pr.esm(s), dtype=torch.float32) for s in syms]).to(dev)
    PRESn = np.zeros((K, LMAX, 32), np.float32); MASKn = np.ones((K, LMAX), bool)
    for i, s in enumerate(syms):
        a = np.asarray(z[s], np.float32)[:LMAX]; PRESn[i, :len(a)] = a; MASKn[i, :len(a)] = False
    PRES = torch.tensor(PRESn).to(dev); MASK = torch.tensor(MASKn).to(dev)
    OE_t = torch.tensor(OE).to(dev); OC_t = torch.tensor(OC).to(dev)

    def run_arch(kind, seed=0):
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        head = (FiLMProfile(dp=E.shape[1]) if kind == "film" else PerresProfile(dp=32)).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
        order = tr.copy()
        for ep in range(epochs):
            rng.shuffle(order)
            for i in range(0, len(order), 64):
                b = order[i:i + 64]; wi = b[:, 0]; ki = b[:, 1]
                Hb = Fb_t[wi]; oe = OE_t[wi, ki]; oc = OC_t[wi, ki]
                if kind == "film":
                    t, c, mix = head(Hb, E[ki])
                else:
                    t, c, mix = head(Hb, PRES[ki], MASK[ki])
                opt.zero_grad(); nll_profile(t, c, mix, oe, oc).backward(); opt.step()
        head.eval()

        def eval_map(kmap):
            per = {k: [] for k in range(K)}
            with torch.no_grad():
                for i in range(0, len(te), 256):
                    b = te[i:i + 256]; wi = b[:, 0]; ki = b[:, 1]; kk = kmap[ki]
                    Hb = Fb_t[wi]
                    if kind == "film":
                        t, _, _ = head(Hb, E[kk])
                    else:
                        t, _, _ = head(Hb, PRES[kk], MASK[kk])
                    pt = torch.softmax(t, 1).cpu().numpy()
                    obs = OE_t[wi, ki].cpu().numpy()
                    for j in range(len(b)):
                        o = obs[j]
                        if o.sum() > 0:
                            per[int(ki[j])].append(pearson(pt[j], o / o.sum()))
            return per
        real = eval_map(np.arange(K)); shuf = eval_map(derangement(K, rng)); fam = eval_map(within_family_perm(fam_ids, rng))
        return real, shuf, fam

    results = {}
    archs = ["film", "perres"] if arch == "both" else [arch]
    for kind in archs:
        real, shuf, fam = run_arch(kind)
        rows = []
        for k in range(K):
            if len(real[k]) >= 5:
                rows.append({"rbp": syms[k], "family": fam_lab[k], "n": len(real[k]),
                             "pearson_real": float(np.mean(real[k])),
                             "pearson_shuf": float(np.mean(shuf[k])) if shuf[k] else float("nan"),
                             "pearson_fam": float(np.mean(fam[k])) if fam[k] else float("nan")})
        rr = np.array([r["pearson_real"] for r in rows]); ss = np.array([r["pearson_shuf"] for r in rows])
        ff = np.array([r["pearson_fam"] for r in rows])
        gap = np.nanmean(rr - ss); gapf = np.nanmean(rr - ff)
        print(f"  {kind:7}: profile Pearson real {np.nanmean(rr):.3f}  shuf {np.nanmean(ss):.3f}  "
              f"gap_der {gap:+.3f}  gap_fam {gapf:+.3f}  (n_rbp {len(rows)})", flush=True)
        results[kind] = {"real": float(np.nanmean(rr)), "shuf": float(np.nanmean(ss)), "fam": float(np.nanmean(ff)),
                         "gap_der": float(gap), "gap_fam": float(gapf), "rows": rows}

    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    (o / "m2_profile.json").write_text(json.dumps(
        {"cell": cell, "K": K, "n_windows": M, "split": split_kind,
         "n_pairs_train": int(len(tr)), "n_pairs_test": int(len(te)),
         "held_chrom": sorted(HELD_CHROM), "archs": results}, indent=1))
    print("wrote m2_profile.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
