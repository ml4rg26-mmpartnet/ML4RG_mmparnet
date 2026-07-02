"""Exp1 (decisive gate): split the M2 zero-shot protein gap into COARSE-envelope vs FINE single-nt components.

The headline M2 metric is already a within-window shape Pearson(predicted-softmax, observed-shape). But a
within-window correlation can be carried by the COARSE envelope (roughly where the broad peak sits -- close to
the M1 'does it bind here' signal) rather than the FINE single-nt placement that is the genuinely nt-resolution,
CORAL-structurally-impossible claim. We Gaussian-decompose each window's shape into coarse (smoothed) + fine
(residual) and recompute the protein-minus-shuffle gap on each. If the FINE gap is > 0 with a paired sign test
across held-out RBPs, the nt-resolution zero-shot thesis is load-bearing.

Reads mmpartnet_out/m2_dump_<tag>.npz (from m2_profile M2_DUMP=1). CPU-only.
  python -m mmpartnet.experiments.m2_decompose <tag> [sigma]
"""
from __future__ import annotations
import sys, os, json
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from mmpartnet import config


def gsmooth(x, sigma):
    """1D Gaussian smoothing via numpy convolution (no scipy dependency), reflect padding."""
    r = int(max(1, round(3 * sigma))); xs = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (xs / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, r, mode="reflect")
    return np.convolve(xp, k, mode="valid")


def shp_pearson(p, q):
    a = p - p.mean(); b = q - q.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def components(pt, obs, sigma):
    """Return full / coarse / fine shape-Pearson of predicted shape pt vs observed shape (obs normalized)."""
    osh = obs / (obs.sum() + 1e-9)
    pc = gsmooth(pt, sigma); oc = gsmooth(osh, sigma)
    pf = pt - pc; of_ = osh - oc
    return shp_pearson(pt, osh), shp_pearson(pc, oc), shp_pearson(pf, of_)


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = d[~np.isnan(d)]
    if len(d) == 0:
        return [float("nan"), float("nan")]
    bs = [np.mean(rng.choice(d, len(d), True)) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def sign_test(d):
    d = d[~np.isnan(d)]; n = len(d); npos = int((d > 0).sum()); bp = wp = float("nan")
    try:
        from scipy import stats as ss
        bp = float(ss.binomtest(npos, n, 0.5).pvalue) if n else float("nan")
        wp = float(ss.wilcoxon(d).pvalue) if n >= 6 and np.any(d != 0) else float("nan")
    except Exception:
        if n:  # normal approx fallback
            from math import erf, sqrt
            z = (npos - n / 2) / (sqrt(n) / 2 + 1e-9); bp = float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))
    return npos, n, bp, wp


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "zeroshot_hepg2"
    sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    out = config.REALDATA / "mmpartnet_out"
    z = np.load(out / f"m2_dump_{tag}.npz", allow_pickle=True)
    rbp = z["rbp"]; syms = list(z["syms"]); fam = list(z["fam"])
    obs = z["obs"].astype(np.float32); ptr = z["pt_real"].astype(np.float32)
    pts = z["pt_shuf"].astype(np.float32); ptf = z["pt_fam"].astype(np.float32)
    comps = ("full", "coarse", "fine")
    # per-window components for each protein condition
    per = {k: {c: {"real": [], "shuf": [], "fam": []} for c in comps} for k in set(rbp.tolist())}
    for i in range(len(rbp)):
        o = obs[i]
        if o.sum() <= 0:
            continue
        cr = components(ptr[i], o, sigma); cs = components(pts[i], o, sigma); cf = components(ptf[i], o, sigma)
        for ci, c in enumerate(comps):
            per[int(rbp[i])][c]["real"].append(cr[ci]); per[int(rbp[i])][c]["shuf"].append(cs[ci]); per[int(rbp[i])][c]["fam"].append(cf[ci])
    # per-RBP means -> gaps; require >=5 windows
    rows = []
    for k, d in per.items():
        if len(d["full"]["real"]) < 5:
            continue
        row = {"rbp": syms[k] if k < len(syms) else str(k), "family": fam[k] if k < len(fam) else "?",
               "n_win": len(d["full"]["real"])}
        for c in comps:
            r = float(np.mean(d[c]["real"])); s = float(np.mean(d[c]["shuf"])); f = float(np.mean(d[c]["fam"]))
            row[f"{c}_real"] = r; row[f"{c}_shuf"] = s; row[f"{c}_fam"] = f
            row[f"{c}_gap_shuf"] = r - s; row[f"{c}_gap_fam"] = r - f
        rows.append(row)
    summary = {"tag": tag, "sigma": sigma, "n_rbp": len(rows)}
    for c in comps:
        gs = np.array([r[f"{c}_gap_shuf"] for r in rows]); gf = np.array([r[f"{c}_gap_fam"] for r in rows])
        npos, n, bp, wp = sign_test(gs)
        summary[c] = {"mean_real": float(np.mean([r[f"{c}_real"] for r in rows])),
                      "gap_shuf": float(np.mean(gs)), "gap_shuf_ci": boot_ci(gs),
                      "gap_fam": float(np.mean(gf)), "gap_fam_ci": boot_ci(gf),
                      "n_beat": npos, "n": n, "binom_p": bp, "wilcoxon_p": wp}
        print(f"  {c:7} gap_vs_shuffle {np.mean(gs):+.4f} CI[{summary[c]['gap_shuf_ci'][0]:+.4f},{summary[c]['gap_shuf_ci'][1]:+.4f}] "
              f"| vs within-family {np.mean(gf):+.4f} | {npos}/{n} p={bp:.1e}")
    (out / f"m2_decompose_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"wrote m2_decompose_{tag}.json (n_rbp={len(rows)})")


if __name__ == "__main__":
    main()
