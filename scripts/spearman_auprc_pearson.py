"""Cross-task difficulty: do the binary and profile tasks agree on which RBPs are hard?

Correlates per-RBP binary AUPRC (from mmpartnet_out/binding_fair.json, the K=68
per-residue panel that backs Figure 1) with per-RBP profile Pearson (from
mmpartnet_out/m2_profile.json, the HepG2 per-residue M2 evaluation), over the RBPs
present in both. Writes mmpartnet_out/spearman_auprc_pearson.json and
docs/img/spearman_auprc_pearson.svg, and prints the summary quoted in the report.

Dependency-light: numpy + matplotlib only (no scipy). The Spearman p-value uses the
Student-t approximation via a pure-python regularized incomplete beta.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HEAD = "perres"   # per-residue head, apples-to-apples on both tasks
CELL = "HepG2"    # m2_profile.json is HepG2

# ---- load per-RBP metrics -------------------------------------------------
bf = json.loads((REPO / "mmpartnet_out" / "binding_fair.json").read_text())
prof = json.loads((REPO / "mmpartnet_out" / "m2_profile.json").read_text())

# binary AUPRC per RBP (mean over that RBP's tracks), from the Figure-1 panel
auprc_rows = defaultdict(list)
for r in bf["methods"][HEAD]["rows"]:
    auprc_rows[r["rbp"]].append(r["real"])
auprc = {k: float(np.mean(v)) for k, v in auprc_rows.items()}

# profile Pearson per RBP (HepG2, per-residue M2 evaluation)
pearson = {r["rbp"]: float(r["pearson_real"]) for r in prof["archs"][HEAD]["rows"]}

common = sorted(set(auprc) & set(pearson))
xs = np.array([auprc[k] for k in common])
ys = np.array([pearson[k] for k in common])
n = len(common)


# ---- Spearman rho + t-based p-value (no scipy) ----------------------------
def _avg_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    df = len(a) - 2
    t = rho * math.sqrt(df / max(1e-12, 1.0 - rho * rho))
    p = _betai(df / 2.0, 0.5, df / (df + t * t))   # two-sided
    return rho, p


rho, p = spearman(xs, ys)
r_pearson = float(np.corrcoef(xs, ys)[0, 1])
amed, pmed = float(np.median(xs)), float(np.median(ys))

hard_both = [k for k in common if auprc[k] < amed and pearson[k] < pmed]
good_both = [k for k in common if auprc[k] > amed and pearson[k] > pmed]
disagree = [k for k in common if (auprc[k] < amed) != (pearson[k] < pmed)]

# the two clearest off-diagonal cases quoted in the report
by_gap = sorted(common, key=lambda k: (auprc[k] - amed) - (pearson[k] - pmed))
binary_hard_profile_easy = by_gap[0]
binary_easy_profile_hard = by_gap[-1]

# ---- write result JSON (committed convention: mmpartnet_out/*.json) -------
result = {
    "analysis": "spearman_auprc_vs_profile_pearson",
    "question": "are poorly-modelled RBPs poorly-modelled in both the binary and profile tasks?",
    "head": HEAD,
    "cell": CELL,
    "sources": {
        "binary_auprc": "mmpartnet_out/binding_fair.json (K=68 per-residue panel; backs Figure 1)",
        "profile_pearson": "mmpartnet_out/m2_profile.json (HepG2 per-residue M2 evaluation)",
    },
    "n_rbp": n,
    "spearman_rho": round(rho, 4),
    "spearman_p": p,
    "pearson_r": round(r_pearson, 4),
    "median_auprc": round(amed, 4),
    "median_pearson": round(pmed, 4),
    "n_hard_in_both": len(hard_both),
    "n_good_in_both": len(good_both),
    "n_disagree": len(disagree),
    "hard_in_both": hard_both,
    "good_in_both": good_both,
    "disagree": disagree,
    "exceptions": {
        binary_hard_profile_easy: {
            "note": "binary-hard, profile-easy",
            "auprc": round(auprc[binary_hard_profile_easy], 4),
            "pearson": round(pearson[binary_hard_profile_easy], 4),
        },
        binary_easy_profile_hard: {
            "note": "binary-easy, profile-hard",
            "auprc": round(auprc[binary_easy_profile_hard], 4),
            "pearson": round(pearson[binary_easy_profile_hard], 4),
        },
    },
    "rows": [
        {"rbp": k, "auprc": round(auprc[k], 4), "pearson": round(pearson[k], 4)}
        for k in common
    ],
}
out_json = REPO / "mmpartnet_out" / "spearman_auprc_pearson.json"
out_json.write_text(json.dumps(result, indent=2))
print(f"wrote {out_json}")

# ---- scatter figure (docs/img convention) ---------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.axvline(amed, color="grey", lw=0.8, ls="--", alpha=0.6)
ax.axhline(pmed, color="grey", lw=0.8, ls="--", alpha=0.6)
ax.scatter(xs, ys, s=28, alpha=0.65, color="C0", edgecolor="white", linewidth=0.4)
for k in (binary_hard_profile_easy, binary_easy_profile_hard):
    ax.scatter([auprc[k]], [pearson[k]], s=42, color="C3", zorder=5)
    ax.annotate(k, (auprc[k], pearson[k]), textcoords="offset points",
                xytext=(6, 4), fontsize=9, color="C3")
ax.set_xlabel("per-RBP binary AUPRC (Figure 1 panel)")
ax.set_ylabel("per-RBP profile Pearson (HepG2, per-residue)")
ax.set_title(f"Hard RBPs are hard in both tasks\n"
             f"Spearman ρ = {rho:.2f}, p = {p:.1e}  (n = {n} RBPs)")
ax.grid(alpha=0.3)
plt.tight_layout()
out_svg = REPO / "docs" / "img" / "spearman_auprc_pearson.svg"
out_svg.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_svg, bbox_inches="tight")
print(f"wrote {out_svg}")

# ---- console summary (matches the sentence in the report) ------------------
print("\nsummary:")
print(f"  n = {n} RBPs ({CELL}, {HEAD} head)")
print(f"  Spearman rho = {rho:.3f}   p = {p:.2e}")
print(f"  Pearson  r   = {r_pearson:.3f}")
print(f"  median AUPRC = {amed:.3f}   median Pearson = {pmed:.3f}")
print(f"  hard in both = {len(hard_both)}   good in both = {len(good_both)}   disagree = {len(disagree)}")
print(f"  binary-hard / profile-easy: {binary_hard_profile_easy} "
      f"(AUPRC {auprc[binary_hard_profile_easy]:.2f}, Pearson {pearson[binary_hard_profile_easy]:.2f})")
print(f"  binary-easy / profile-hard: {binary_easy_profile_hard} "
      f"(AUPRC {auprc[binary_easy_profile_hard]:.2f}, Pearson {pearson[binary_easy_profile_hard]:.2f})")
