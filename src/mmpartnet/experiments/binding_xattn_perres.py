"""Per-residue (TFBindFormer/CORAL-faithful) cross-attention vs pooled, on the rigorous binding-auPRC CV --
retrying the prior per-residue+domain idea (per_residue.json, which only had a weak MRR metric) inside the
framework of binding_mechanism. Three conditions on IDENTICAL splits/seeds, to separate per-residue STRUCTURE
from embedding DIMENSION:

  pooled32  CrossAttnHead, protein = mean over the (L,32) per-residue embedding -> a single 32-d query
  perres    BiCrossAttnHead, RNA positions attend over the protein's (L,32) residue tokens (K/V) + mask
  pooledESM CrossAttnHead, protein = full pooled ESM (the richer-dim reference line)

pooled32 vs perres isolates the per-residue mechanism (same 32-d source); pooledESM shows where dimension
sits. INTERPRETABILITY: the per-residue head's protein-residue attention (B,Prna,Lp) -> per-residue mass;
we test DOMAIN ENRICHMENT = mean attention inside annotated RRM/KH domains vs overall (>1 = the model reads
the RNA through its RNA-binding domains), with a same-size random-window control.

  python -m mmpartnet.experiments.binding_xattn_perres [scheme] [k_seeds] [n_train] [n_test]
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
from mmpartnet.models.cross_attn_head import CrossAttnHead, BiCrossAttnHead
from mmpartnet.models.parnet import load_parnet
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.experiments.binding_eval import average_precision, read_rbp
from mmpartnet.experiments.binding_head import load_split

LWIN = 600
NPOS = 64
LMAX = 512        # cap protein residues (covers RRM/KH domains of the canonical RBPs)


def feats_seq(m, seqs, bs=128):
    out = []
    for i in range(0, len(seqs), bs):
        x = batch_onehot(seqs[i:i + bs], device=m.device)
        with torch.no_grad():
            h = F.adaptive_avg_pool1d(m.body_feats(x), NPOS)
        out.append(h.transpose(1, 2).cpu())
    return torch.cat(out)


def load_perres(bundle, syms):
    """Return PRES (K,LMAX,32), MASK (K,LMAX) bool (True=pad), and per-sym residue length."""
    z = np.load(f"{bundle}/perres32.npz", allow_pickle=True)
    K = len(syms); dpr = 32
    PRES = np.zeros((K, LMAX, dpr), np.float32); MASK = np.ones((K, LMAX), bool); LEN = np.zeros(K, int)
    for i, s in enumerate(syms):
        if s in z.files:
            a = np.asarray(z[s], np.float32)[:LMAX]
            PRES[i, :len(a)] = a; MASK[i, :len(a)] = False; LEN[i] = len(a)
    return PRES, MASK, LEN


def aps_pooled(head, Fte, te_y, ti_keep, Emap, K, dev):
    out = np.full(K, np.nan)
    with torch.no_grad():
        for k in range(K):
            ss = []
            for i in range(0, len(Fte), 512):
                Hb = Fte[i:i + 512]
                ss.append(torch.sigmoid(head(Hb, Emap[k].expand(len(Hb), -1))).cpu().numpy())
            s = np.concatenate(ss); y = (te_y[:, ti_keep[k]] > 0).astype(float)
            if y.sum() >= 5:
                out[k] = average_precision(s, y)
    return out


def aps_perres(head, Fte, te_y, ti_keep, PRES, MASK, K, perm, dev):
    out = np.full(K, np.nan)
    with torch.no_grad():
        for k in range(K):
            kk = perm[k]
            Pk = PRES[kk:kk + 1]; Mk = MASK[kk:kk + 1]
            ss = []
            for i in range(0, len(Fte), 256):
                Hb = Fte[i:i + 256]
                ss.append(torch.sigmoid(head(Hb, Pk.expand(len(Hb), -1, -1), Mk.expand(len(Hb), -1))).cpu().numpy())
            s = np.concatenate(ss); y = (te_y[:, ti_keep[k]] > 0).astype(float)
            if y.sum() >= 5:
                out[k] = average_precision(s, y)
    return out


def train_pooled(Ftr, Ytr, Fte, te_y, ti_keep, E, K, seed, dev):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    head = CrossAttnHead(d_model=Ftr.shape[2], dp=E.shape[1], heads=4, layers=2).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=5e-4, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
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
    return head, aps_pooled(head, Fte, te_y, ti_keep, E, K, dev), \
        aps_pooled(head, Fte, te_y, ti_keep, E[rng.permutation(K)], K, dev)


def train_perres(Ftr, Ytr, Fte, te_y, ti_keep, PRES, MASK, K, seed, dev):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    head = BiCrossAttnHead(d_model=Ftr.shape[2], dp=PRES.shape[2], heads=4, layers=2).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=5e-4, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
    pos = torch.nonzero(Ytr > 0, as_tuple=False).cpu().numpy()
    for ep in range(10):
        rng.shuffle(pos)
        for i in range(0, len(pos), 128):
            pb = pos[i:i + 128]
            wp = torch.tensor(pb[:, 0], device=dev); kp = torch.tensor(pb[:, 1], device=dev)
            kn = torch.tensor(rng.integers(0, K, size=len(pb)), device=dev)
            kall = torch.cat([kp, kn])
            Hb = torch.cat([Ftr[wp], Ftr[wp]])
            Pb = PRES[kall]; Mb = MASK[kall]
            yy = torch.cat([torch.ones(len(pb), device=dev), Ytr[wp, kn]])
            opt.zero_grad(); bce(head(Hb, Pb, Mb), yy).backward(); opt.step()
    head.eval()
    real = aps_perres(head, Fte, te_y, ti_keep, PRES, MASK, K, np.arange(K), dev)
    shuf = aps_perres(head, Fte, te_y, ti_keep, PRES, MASK, K, rng.permutation(K), dev)
    return head, real, shuf


def summarize(REAL, SH):
    rmean = np.nanmean(REAL, 0); smean = np.nanmean(SH, 0)
    gseed = np.nanmean(REAL, 1) - np.nanmean(SH, 1)
    valid = ~np.isnan(rmean) & ~np.isnan(smean); diff = (rmean - smean)[valid]
    rng = np.random.default_rng(0)
    boot = np.array([np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {"real": float(np.nanmean(rmean)), "shuffle": float(np.nanmean(smean)),
            "gap": float(gseed.mean()), "gap_std": float(gseed.std()), "gap_ci95": ci,
            "n_better": int((diff > 0).sum()), "n_rbp": int(valid.sum()),
            "rmean": rmean.tolist(), "smean": smean.tolist()}


def domain_enrichment(head, Fte, te_y, ti_keep, PRES, MASK, LEN, keep, domains, dev, max_rbp=12, max_win=12):
    """For RBPs with annotated RRM/KH domains: fraction of protein-residue attention mass inside domains vs a
    same-size random-window control. enrichment = mean(attn in domain) / mean(attn overall)."""
    rbp_dom = {}
    for ti, r, c in keep:
        dl = domains.get(r) or []
        spans = [(int(s), int(e)) for s, e, lab in dl if any(t in str(lab) for t in ("RRM", "KH"))]
        if spans:
            rbp_dom[r] = spans
    rows = []
    rng = np.random.default_rng(0)
    idx = {r: i for i, (ti, r, c) in enumerate(keep)}
    chosen = [r for ti, r, c in keep if r in rbp_dom][:max_rbp]
    for r in chosen:
        k = idx[r]; ti = ti_keep[k]
        widx = np.nonzero(te_y[:, ti] > 0)[0][:max_win]
        if len(widx) == 0:
            continue
        L = int(LEN[k])
        dommask = np.zeros(LMAX, bool)
        for s, e in rbp_dom[r]:
            dommask[max(0, s):min(LMAX, e)] = True
        dommask &= ~MASK[k].cpu().numpy()
        ndom = int(dommask.sum())
        if ndom == 0 or L < 5:
            continue
        Pk = PRES[k:k + 1]; Mk = MASK[k:k + 1]
        att = np.zeros(LMAX)
        for wi in widx:
            Hb = Fte[int(wi):int(wi) + 1]
            with torch.no_grad():
                _, a = head(Hb, Pk, Mk, return_attn=True)          # (1, Prna, Lp)
            att += a[0].mean(0).cpu().numpy()                       # mean over RNA positions
        att = att[:L]; att = att / (att.sum() + 1e-9)
        dm = dommask[:L]
        in_dom = att[dm].mean() if dm.any() else np.nan
        overall = att.mean()
        enr = float(in_dom / (overall + 1e-12))
        # random control: same #domain residues drawn at random
        ctrl = []
        for _ in range(50):
            rp = rng.choice(L, min(ndom, L), replace=False)
            ctrl.append(att[rp].mean() / (overall + 1e-12))
        rows.append({"rbp": r, "n_dom_res": ndom, "prot_len": L, "enrichment": enr,
                     "ctrl_mean": float(np.mean(ctrl)), "ctrl_std": float(np.std(ctrl)),
                     "domains": rbp_dom[r]})
    return rows


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
    keep = [(ti, r, c) for ti, (r, c) in enumerate(tracks) if pr.esm(r) is not None and r in have]
    K = len(keep); ti_keep = [ti for ti, r, c in keep]; syms = [r for ti, r, c in keep]
    m = load_parnet(device=dev)
    print(f"binding_xattn_perres [{scheme}] K_rbp={K} (have perres) seeds={kseeds} dev={dev}", flush=True)
    tr_seq, tr_y = load_split(base, "train", n_train); te_seq, te_y = load_split(base, "test", n_test)
    Ftr = feats_seq(m, tr_seq).to(dev); Fte = feats_seq(m, te_seq).to(dev)
    Ytr = torch.tensor((tr_y[:, ti_keep] > 0).astype(np.float32)).to(dev)
    Eesm = torch.stack([torch.tensor(pr.esm(r), dtype=torch.float32) for r in syms]).to(dev)
    PRESn, MASKn, LEN = load_perres(bundle, syms)
    PRES = torch.tensor(PRESn).to(dev); MASK = torch.tensor(MASKn).to(dev)
    cnt = (~MASK).sum(1, keepdim=True).clamp(min=1)
    E32 = (PRES.sum(1) / cnt).to(dev)                               # mean-pooled 32-d query (same source)

    results = {}; perres_head = None
    for cond in ("pooled32", "pooledESM", "perres"):
        REAL = np.zeros((kseeds, K)); SH = np.zeros((kseeds, K))
        for s in range(kseeds):
            if cond == "perres":
                h, REAL[s], SH[s] = train_perres(Ftr, Ytr, Fte, te_y, ti_keep, PRES, MASK, K, s, dev)
                if s == 0:
                    perres_head = h
            else:
                Eu = E32 if cond == "pooled32" else Eesm
                _, REAL[s], SH[s] = train_pooled(Ftr, Ytr, Fte, te_y, ti_keep, Eu, K, s, dev)
        res = summarize(REAL, SH); results[cond] = res
        print(f"  {cond:9}: real {res['real']:.3f} shuffle {res['shuffle']:.3f} gap {res['gap']:+.3f}"
              f"+/-{res['gap_std']:.3f} better {res['n_better']}/{res['n_rbp']} "
              f"CI[{res['gap_ci95'][0]:+.3f},{res['gap_ci95'][1]:+.3f}]", flush=True)

    # interpretability: domain enrichment of the per-residue head's protein attention
    dom_rows = []
    dpath = f"{bundle}/domains_150.json"
    if perres_head is not None and os.path.exists(dpath):
        domains = json.load(open(dpath))
        dom_rows = domain_enrichment(perres_head, Fte, te_y, ti_keep, PRES, MASK, LEN, keep, domains, dev)
        if dom_rows:
            enr = np.array([r["enrichment"] for r in dom_rows]); ctl = np.array([r["ctrl_mean"] for r in dom_rows])
            print(f"\n[domain attention] mean enrichment {enr.mean():.2f}x (control {ctl.mean():.2f}x) "
                  f"over {len(dom_rows)} RBPs; enriched>control in {int((enr>ctl).sum())}/{len(dom_rows)}", flush=True)

    o = config.REALDATA / "mmpartnet_out"; o.mkdir(parents=True, exist_ok=True)
    (o / "binding_xattn_perres.json").write_text(json.dumps(
        {"scheme": scheme, "seeds": kseeds, "n_train": len(tr_seq), "n_test": len(te_seq), "K": K,
         "rbps": syms, "conditions": results, "domain_rows": dom_rows}, indent=1))
    print("wrote binding_xattn_perres.json\ndone.", flush=True)


if __name__ == "__main__":
    main()
