"""BioPWM (Ch206), UPGRADED to the contract: the protein writes its own motif detector via a PER-RESIDUE
RBD-weighted RECOGNITION-CODE generator (not a pooled MLP). A learned set of K position-queries Qp attends over
the protein's per-residue tokens (RBD residues win the attention); the attended context is mapped through a SHARED
amino-acid->base code T to a width-K PWM (log-odds). The PWM scans the RAW RNA one-hot (Stormo log-odds + Foat
logsumexp soft-occupancy) -> binding logit. This is the SPECIFICITY track and is LEAKAGE-FREE (no PARNET).

Optional combined mode (BIAS=1): add a protein-agnostic PARNET-feature bias logit (the actual lab PARNET
representation). The protein-vs-shuffle GAP stays leakage-free because the shared bias cancels in the difference,
while absolute prediction benefits from PARNET -- matching the lab combined_biopwm design.

Controls: real protein vs CROSS-FAMILY (derangement) shuffle AND WITHIN-FAMILY shuffle, 5 seeds, leave-out-RBP
zero-shot option. Saves the generated width-K PWMs (the explicit interpretable latent) for ATtRACT (pwm.txt)
recovery. Lesson kept (Ch206): logsumexp pool, never a saturating max.

  python -m mmpartnet.experiments.binding_biopwm [scheme] [seeds] [n_train] [n_test] [split]   # split: indist|rbp
  env: GEN=recog|mlp (default recog)   BIAS=1 (combine PARNET bias)   DRES (per-residue dim source)
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
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.models.parnet import load_parnet
from mmpartnet.experiments.binding_eval import average_precision, read_rbp
from mmpartnet.experiments.binding_head import load_split
from mmpartnet.experiments import eval_controls as EC

LOGB = float(np.log(0.25))
KW = 8                 # motif width
LMAX = 384             # protein-residue cap
GEN = os.environ.get("GEN", "recog")
USE_BIAS = os.environ.get("BIAS", "0") == "1"


class BioPWMHead(nn.Module):
    """Per-residue recognition-code generator -> width-KW PWM -> scan RNA one-hot -> binding logit (leakage-free).
    Optional PARNET-feature bias (protein-agnostic; cancels in the real-vs-shuffle gap)."""
    def __init__(self, dres=32, dr_bias=1024, beta=4.0, hidden=128, use_bias=False):
        super().__init__()
        self.beta = beta; self.use_bias = use_bias
        if GEN == "recog":
            self.Qp = nn.Parameter(torch.randn(KW, dres) * 0.1)      # KW position-queries over residues
            self.T = nn.Parameter(torch.randn(dres, 4) * 0.05)       # shared aa->base recognition code
        else:
            self.gen = nn.Sequential(nn.Linear(dres, hidden), nn.GELU(), nn.Linear(hidden, KW * 4))
        self.scale = nn.Parameter(torch.tensor(1.0)); self.b0 = nn.Parameter(torch.zeros(1))
        if use_bias:
            self.bias_head = nn.Sequential(nn.LayerNorm(dr_bias), nn.Linear(dr_bias, hidden), nn.GELU(),
                                           nn.Dropout(0.1), nn.Linear(hidden, 1))

    def pwm_logodds(self, Ep, M):                                    # Ep:(B,R,dres) M:(B,R) True=valid -> W:(B,KW,4)
        if GEN == "recog":
            att = torch.einsum("kd,brd->bkr", self.Qp, Ep) / (Ep.shape[-1] ** 0.5)
            att = att.masked_fill(~M[:, None, :], -1e9)
            ctx = torch.einsum("bkr,brd->bkd", torch.softmax(att, -1), Ep)
            return torch.einsum("bkd,de->bke", ctx, self.T)         # (B,KW,4) log-odds
        e = (Ep * M.unsqueeze(-1)).sum(1) / M.sum(1, keepdim=True).clamp(min=1)   # masked-mean pooled
        P = torch.softmax(self.gen(e).view(-1, KW, 4), -1)
        return torch.log(P + 1e-6) - LOGB

    def pwm(self, Ep, M):
        return torch.softmax(self.pwm_logodds(Ep, M), dim=-1)       # proper PWM (B,KW,4) for ATtRACT recovery

    def forward(self, X, Ep, M, Hp=None):                           # X:(B,4,L) one-hot, Ep:(B,R,dres), M:(B,R)
        W = self.pwm_logodds(Ep, M)
        Xw = X.permute(0, 2, 1).unfold(1, KW, 1).permute(0, 1, 3, 2)            # (B, Lpos, KW, 4)
        s = torch.einsum("blkc,bkc->bl", Xw, W)                     # (B, Lpos) occupancy on raw RNA
        aff = torch.logsumexp(self.beta * s, dim=1) / self.beta     # (B,) Foat soft-occupancy
        logit = self.scale * aff + self.b0
        if self.use_bias and Hp is not None:
            logit = logit + self.bias_head(Hp).squeeze(-1)          # protein-agnostic PARNET bias (cancels in gap)
        return logit


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "pureclip"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 25000
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 15000
    split = sys.argv[5] if len(sys.argv) > 5 else "indist"
    base = os.path.expanduser(f"~/ml4rg_data/binding/{scheme}")
    bundle = os.environ.get("ML4RG_DATA", os.path.expanduser("~/ml4rg_data"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = read_rbp(f"{base}/rbp_cts.tsv"); pr = embeddings.ProteinRep()
    z = np.load(f"{bundle}/perres32.npz", allow_pickle=True); have = set(z.files)
    keep = [(ti, r, c) for ti, (r, c) in enumerate(tracks) if pr.esm(r) is not None and r in have]
    K = len(keep); ti_keep = [ti for ti, r, c in keep]; syms = [r for ti, r, c in keep]
    fam_map = cohort.attract_families(syms)
    fam_lab = [(fam_map[r][0] if r in fam_map and fam_map[r] else "other") for r in syms]
    fam_ids = np.array([sorted(set(fam_lab)).index(f) for f in fam_lab])
    dres = int(np.asarray(z[syms[0]]).shape[1])
    PRESn = np.zeros((K, LMAX, dres), np.float32); MASKn = np.zeros((K, LMAX), bool)   # MASK True=valid
    for i, s in enumerate(syms):
        a = np.asarray(z[s], np.float32)[:LMAX]; PRESn[i, :len(a)] = a; MASKn[i, :len(a)] = True
    PRES = torch.tensor(PRESn).to(dev); MASK = torch.tensor(MASKn).to(dev)
    tr_seq, tr_y = load_split(base, "train", n_train); te_seq, te_y = load_split(base, "test", n_test)
    Ytr = torch.tensor((tr_y[:, ti_keep] > 0).astype(np.float32)).to(dev)
    # PARNET bias features (only if combined)
    Htr = Hte = None
    if USE_BIAS:
        m = load_parnet(device=dev)
        Htr = EC.pooled(EC.feats_pos(m, tr_seq)).to(dev); Hte = EC.pooled(EC.feats_pos(m, te_seq)).to(dev)
    print(f"binding_biopwm [{scheme}] K={K} seeds={seeds} split={split} GEN={GEN} BIAS={USE_BIAS} dev={dev} "
          f"(specificity track LEAKAGE-FREE)", flush=True)

    rng0 = np.random.default_rng(0)
    held = set(rng0.choice(K, max(1, int(0.3 * K)), replace=False).tolist()) if split == "rbp" else set()
    train_cols = np.array([k for k in range(K) if k not in held]); eval_cols = sorted(held) if split == "rbp" else list(range(K))

    def onehot(seqs, idx):
        return batch_onehot([seqs[i] for i in idx], device=dev)

    def run_seed(s):
        rng = np.random.default_rng(s); torch.manual_seed(s)
        head = BioPWMHead(dres=dres, dr_bias=(Htr.shape[1] if USE_BIAS else 1024), use_bias=USE_BIAS).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=5e-4, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
        Ytr_m = Ytr.clone()
        if held:
            Ytr_m[:, sorted(held)] = 0
        pos = torch.nonzero(Ytr_m > 0, as_tuple=False).cpu().numpy()
        for ep in range(12):
            rng.shuffle(pos)
            for i in range(0, len(pos), 128):
                pb = pos[i:i + 128]; nb = len(pb); wp = pb[:, 0]; kp = pb[:, 1]
                kn = train_cols[rng.integers(0, len(train_cols), size=nb)]
                X = torch.cat([onehot(tr_seq, wp), onehot(tr_seq, wp)])
                kk = torch.tensor(np.concatenate([kp, kn]), device=dev)
                wt = torch.tensor(np.concatenate([wp, wp]), device=dev)
                Hp = torch.cat([Htr[torch.tensor(wp, device=dev)], Htr[torch.tensor(wp, device=dev)]]) if USE_BIAS else None
                yy = torch.cat([torch.ones(nb, device=dev), Ytr[torch.tensor(wp, device=dev), torch.tensor(kn, device=dev)]])
                opt.zero_grad(); bce(head(X, PRES[kk], MASK[kk], Hp), yy).backward(); opt.step()
        head.eval()

        def aps(kmap):
            out = np.full(K, np.nan)
            with torch.no_grad():
                for k in eval_cols:
                    j = int(kmap[k]); ss = []
                    for i in range(0, len(te_seq), 256):
                        idx = list(range(i, min(i + 256, len(te_seq)))); n = len(idx)
                        Hp = Hte[i:i + 256] if USE_BIAS else None
                        ss.append(torch.sigmoid(head(onehot(te_seq, idx), PRES[j:j + 1].expand(n, -1, -1),
                                                       MASK[j:j + 1].expand(n, -1), Hp)).cpu().numpy())
                    sc = np.concatenate(ss); y = (te_y[:, ti_keep[k]] > 0).astype(float)
                    if y.sum() >= 5:
                        out[k] = average_precision(sc, y)
            return out
        real = aps(np.arange(K)); shuf = aps(EC.derangement(K, rng)); fam = aps(EC.within_family_perm(fam_ids, rng))
        pwm = head.pwm(PRES, MASK).detach().cpu().numpy().astype(np.float16) if s == 0 else None
        return real, shuf, fam, pwm

    REAL = np.full((seeds, K), np.nan); SHUF = np.full((seeds, K), np.nan); FAM = np.full((seeds, K), np.nan); PWM = None
    for s in range(seeds):
        REAL[s], SHUF[s], FAM[s], p = run_seed(s)
        if p is not None:
            PWM = p
    rm = np.nanmean(REAL, 0); sm = np.nanmean(SHUF, 0); fm = np.nanmean(FAM, 0)
    v = ~np.isnan(rm) & ~np.isnan(sm); gap = (rm - sm)[v]; vf = ~np.isnan(rm) & ~np.isnan(fm); gapf = (rm - fm)[vf]
    npos, n, bp, wp = EC.sign_test(gap); ci = EC.boot_ci(gap)
    nposf, nf, bpf, wpf = EC.sign_test(gapf); cif = EC.boot_ci(gapf)
    print(f"BioPWM[{GEN},bias={USE_BIAS}] real {np.nanmean(rm):.4f} | vs cross-family {np.nanmean(gap):+.4f} "
          f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] {npos}/{n} wilcoxon={wp:.1e} | vs WITHIN-FAMILY {np.nanmean(gapf):+.4f} "
          f"CI[{cif[0]:+.4f},{cif[1]:+.4f}] {nposf}/{nf} wilcoxon={wpf:.1e}", flush=True)
    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    rows = [{"rbp": syms[k], "family": fam_lab[k], "real": float(rm[k]), "shuffle": float(sm[k]), "fam": float(fm[k]),
             "gap": float(rm[k] - sm[k]), "gap_fam": float(rm[k] - fm[k])} for k in range(K) if v[k]]
    tag = f"biopwm_{GEN}{'_bias' if USE_BIAS else ''}_{split}"
    out = {"scheme": scheme, "split": split, "gen": GEN, "bias": USE_BIAS, "K": K, "eval_n": int(v.sum()),
           "leakage_free": (not USE_BIAS), "real": float(np.nanmean(rm)), "shuffle": float(np.nanmean(sm)),
           "fam": float(np.nanmean(fm)), "gap": float(np.nanmean(gap)), "gap_ci": ci, "n_beat": npos,
           "binom_p": bp, "wilcoxon_p": wp, "gap_fam": float(np.nanmean(gapf)), "gap_fam_ci": cif,
           "n_beat_fam": nposf, "wilcoxon_p_fam": wpf, "rows": rows, "rbps": syms}
    (o / f"binding_{tag}.json").write_text(json.dumps(out, indent=1))
    if PWM is not None:
        np.savez_compressed(o / f"{tag}_pwms.npz", pwm=PWM, syms=np.array(syms, object), fam=np.array(fam_lab, object))
        print(f"wrote {tag}_pwms.npz {PWM.shape}", flush=True)
    print(f"wrote binding_{tag}.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
