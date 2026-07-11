"""Per-RBP profile-Pearson distribution across M2 architectures + cell lines.

Companion to scripts/plot_per_rbp_distribution.py (which does the same for
binary AUPRC on the K=68 panel). Reads mmpartnet_out/m2_profile.json and
m2_profile_K562.json — per-RBP profile Pearson computed by the M2 harness —
and produces docs/img/per_rbp_pearson_distribution.svg.

Rows have: pearson_real (model), pearson_shuf (protein-shuffle control),
pearson_fam (within-family shuffle control).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

CELLS = {
    "HepG2": REPO / "mmpartnet_out" / "m2_profile.json",
    "K562":  REPO / "mmpartnet_out" / "m2_profile_K562.json",
}

def load(cell_path: Path) -> dict[str, list[float]]:
    d = json.loads(cell_path.read_text())
    out: dict[str, list[float]] = {}
    for arch, blob in d["archs"].items():
        out[arch] = [r["pearson_real"] for r in blob["rows"]]
    # baseline: protein-shuffle uses same rows from any arch (they share input)
    any_rows = next(iter(d["archs"].values()))["rows"]
    out["Shuffle"] = [r["pearson_shuf"] for r in any_rows]
    return out

series_by_cell = {cell: load(p) for cell, p in CELLS.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (cell, series) in zip(axes, series_by_cell.items()):
    labels = list(series.keys())
    values = [series[k] for k in labels]
    positions = np.arange(len(labels))
    vp = ax.violinplot(values, positions=positions, widths=0.75,
                       showmeans=False, showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.4)
        body.set_edgecolor("black")
    vp["cmedians"].set_color("black")
    for i, v in enumerate(values):
        ax.scatter(np.random.normal(i, 0.06, size=len(v)), v, s=10, alpha=0.5, color="C0")
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=15)
    ax.set_title(f"{cell}  (N={len(next(iter(series.values())))} RBPs)")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("per-RBP profile Pearson")

fig.suptitle("M2 per-RBP profile Pearson — every dot is one RBP", y=1.02)
plt.tight_layout()

out = REPO / "docs" / "img" / "per_rbp_pearson_distribution.svg"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")

# summary numbers for the doc
print("\nsummary (per-RBP profile Pearson):")
for cell, series in series_by_cell.items():
    print(f"\n  {cell}:")
    for k, v in series.items():
        v = np.array(v)
        print(f"    {k:10s}  mean={v.mean():+.4f}  median={np.median(v):+.4f}  "
              f"top-10%={np.quantile(v, 0.9):+.4f}  bottom-10%={np.quantile(v, 0.1):+.4f}")
