"""Recover the PARNET demo's per-RBP profile result on OUR public ENCODE data.

The demo evaluates per-RBP profile **Pearson/Spearman** between PARNET's predicted per-nt profile
(``out['total']``, normalized) and the observed per-nt eCLIP signal, on test tiles with >=10 reads.
We cannot use its gated dataset, but we reproduce its EVALUATION on the genuine ENCODE signal:
take an RBP's real binding windows (our peak BEDs) -> predict the profile with the frozen pretrained
PARNET -> read the observed eCLIP signal from the public ENCODE bigWig (remote) -> the demo's metric.

This is the PRETRAINED baseline the demo reports. Verification: a positive correlation that is
**>> a position-shuffled control** means the pipeline + the real model recover the demo behaviour on
our data. (Head-finetuning is the next step; this establishes the frozen-PARNET baseline first.)

  python -m mmpartnet.experiments.recover_demo_profile --group AQR --cell HepG2 --nwin 40
"""
from __future__ import annotations
import os, sys, json, gzip, argparse
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from mmpartnet import config
from mmpartnet.io import genome, groups
from mmpartnet.process.onehot import batch_onehot
from mmpartnet.models.parnet import load_parnet
from mmpartnet.adapters import eclip_signal
from mmpartnet.adapters import eclip_counts
from mmpartnet.adapters import peaks as peaks_adapter

LWIN = 600
MIN_RC_SUM = 10        # demo PARAMS_MIN_RC_SUM: only score windows with >=10 observed reads


def _pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-9 else np.nan


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return _pearson(rx, ry)


def _center_bump(L, sigma=None):
    """Control 4 baseline predictor: a fixed Gaussian bump centered at L/2. Because the windows are
    peak-centerD, a dumb central enrichment correlates with the observed signal by construction; if
    PARNET barely beats this, the 'signal' is mostly the peak-centering artifact, not learned shape."""
    sigma = sigma or (L / 12.0)
    x = np.arange(L, dtype=float)
    b = np.exp(-0.5 * ((x - (L - 1) / 2.0) / sigma) ** 2)
    s = b.sum()
    return b / s if s > 0 else b


def _peaks(path, max_n, top_by_score=True):
    out = []
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            strand = f[5] if len(f) > 5 and f[5] in "+-" else "+"
            score = float(f[6]) if len(f) > 6 and f[6] not in (".", "") else 0.0
            out.append((f[0], int(f[1]), int(f[2]), strand, score))
    if top_by_score:
        out.sort(key=lambda p: -p[4])
    return out[:max_n]


def recover_rbp(m, rbp, cell, manifest, nwin, target="density", bam_dir=None):
    rec = next((r for r in manifest.get(rbp, []) if r["cell"] == cell), None)
    if rec is None:
        return {"rbp": rbp, "cell": cell, "error": "no manifest record for cell"}
    tr = m.track_index(rbp, cell)
    if tr is None:
        return {"rbp": rbp, "cell": cell, "error": "no PARNET track"}
    if target == "counts":          # Control 1: established 5'-crosslink counts from ENCODE BAMs
        reader, _ctrl, urls = eclip_counts.readers_for(rec["exp"], bam_dir)
        control_exp = urls.get("control_exp")
        if reader is None:
            return {"rbp": rbp, "cell": cell, "error": "no GRCh38 alignment BAMs"}
    else:                           # default: RPM read-density from the bigWig
        urls = eclip_signal.resolve_signal_urls(rec["exp"])
        reader = eclip_signal.SignalReader(urls["plus"], urls["minus"])
        control_exp = urls["control_exp"]
    peaks = _peaks(peaks_adapter.resolve_bed(rec), max_n=nwin * 3)   # portable path; over-sample

    pear, spear, sh_pear, circ_pear, cbump_pear, used = [], [], [], [], [], 0
    cbump = _center_bump(LWIN)                            # fixed center-bias baseline predictor
    for chrom, s, e, strand, _score in peaks:
        if used >= nwin or not reader.has(chrom):
            if used >= nwin:
                break
            continue
        c = (s + e) // 2; ws = c - LWIN // 2
        seq = genome.centered_window(chrom, s, e, strand, LWIN)
        if seq is None:
            continue
        obs = reader.profile(chrom, ws, ws + LWIN, strand)
        if obs is None or obs.sum() < MIN_RC_SUM:
            continue
        x = batch_onehot([seq], device=m.device)
        pred = m.full(x)["total"][0, tr].detach().cpu().numpy()   # softmaxed prob over 600 positions
        obs_p = obs / obs.sum()
        pear.append(_pearson(pred, obs_p))
        spear.append(_spearman(pred, obs_p))
        # --- nulls / baselines (Control 4: established nulls for autocorrelated genomic signal) ---
        rng = np.random.default_rng(used)
        sh = obs_p.copy(); rng.shuffle(sh)
        sh_pear.append(_pearson(pred, sh))               # (weak) per-position i.i.d. shuffle
        k = int(rng.integers(LWIN // 8, LWIN - LWIN // 8))
        circ_pear.append(_pearson(pred, np.roll(obs_p, k)))  # circular shift: PRESERVES autocorrelation
        cbump_pear.append(_pearson(cbump, obs_p))            # dumb center-bump baseline predictor vs obs
        used += 1

    if used == 0:
        return {"rbp": rbp, "cell": cell, "error": "no windows passed the >=10-read filter"}
    return {"rbp": rbp, "cell": cell, "exp": rec["exp"], "control_exp": control_exp,
            "n_windows": used,
            "pearson_mean": float(np.nanmean(pear)), "pearson_std": float(np.nanstd(pear)),
            "spearman_mean": float(np.nanmean(spear)), "spearman_std": float(np.nanstd(spear)),
            "shuffled_pearson_mean": float(np.nanmean(sh_pear)),
            "circular_pearson_mean": float(np.nanmean(circ_pear)),   # autocorr-matched null
            "centerbump_pearson_mean": float(np.nanmean(cbump_pear))}  # center-bias baseline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="AQR", help="RBP group/name (groups.resolve): AQR | spliceosome | <SYM,...>")
    ap.add_argument("--cell", default="HepG2")
    ap.add_argument("--nwin", type=int, default=40)
    ap.add_argument("--target", default="density", choices=["density", "counts"],
                    help="Control 1: 'density' = RPM bigWig (proxy); 'counts' = established 5'-crosslink counts from ENCODE BAMs")
    args = ap.parse_args()

    bam_dir = os.environ.get("ML4RG_BAMS", str(config.DATA / "bams"))
    rbps = groups.resolve(args.group) or [args.group]
    manifest = json.loads((config.DATA / "eclip_manifest.json").read_text())
    rbps = [g for g in rbps if g in manifest]
    print(f"recover demo profile (frozen pretrained PARNET) | group={args.group} cell={args.cell} "
          f"RBPs={rbps} nwin={args.nwin} target={args.target}", flush=True)
    m = load_parnet()

    rows = []
    for g in rbps:
        r = recover_rbp(m, g, args.cell, manifest, args.nwin, target=args.target, bam_dir=bam_dir)
        rows.append(r)
        if "error" in r:
            print(f"  {g:8} {args.cell}: SKIP ({r['error']})", flush=True)
        else:
            print(f"  {g:8} {args.cell}: Pearson={r['pearson_mean']:+.3f}+/-{r['pearson_std']:.3f}  "
                  f"Spearman={r['spearman_mean']:+.3f}  shuf={r['shuffled_pearson_mean']:+.3f}  "
                  f"circ={r['circular_pearson_mean']:+.3f}  cbump={r['centerbump_pearson_mean']:+.3f}  "
                  f"(n={r['n_windows']}, exp={r['exp']})", flush=True)

    ok = [r for r in rows if "error" not in r]
    if ok:
        mp = float(np.mean([r["pearson_mean"] for r in ok]))
        msh = float(np.mean([r["shuffled_pearson_mean"] for r in ok]))
        mc = float(np.mean([r["circular_pearson_mean"] for r in ok]))
        mb = float(np.mean([r["centerbump_pearson_mean"] for r in ok]))
        # Established-null verdict: must beat the AUTOCORR-matched circular null AND the center-bump
        # baseline (not just the trivial i.i.d. shuffle) to be real binding-shape signal.
        real = (mp - mc > 0.05) and (mp - mb > 0.05)
        print(f"\nSUMMARY: mean Pearson={mp:+.3f}  | nulls: shuffle={msh:+.3f} circular={mc:+.3f} "
              f"center-bump={mb:+.3f}", flush=True)
        print(f"  -> Pearson - circular = {mp-mc:+.3f} ; Pearson - center-bump = {mp-mb:+.3f}  "
              f"-> {'REAL SHAPE (beats established nulls)' if real else 'NOT BEYOND center/AUTOCORR — downgrade'}", flush=True)
    out = {"cell": args.cell, "group": args.group, "min_rc_sum": MIN_RC_SUM, "target": args.target, "rows": rows}
    o = config.RESULTS; o.mkdir(parents=True, exist_ok=True)
    fname = "recover_demo_profile.json" if args.target == "density" else f"recover_demo_profile_{args.target}.json"
    (o / fname).write_text(json.dumps(out, indent=1))
    print(f"wrote {o / fname}", flush=True)


if __name__ == "__main__":
    main()
