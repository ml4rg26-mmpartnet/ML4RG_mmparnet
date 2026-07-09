"""Per-RBP AUPRC distribution across the 4 conditioning architectures.

Reads mmpartnet_out/binding_fair.json (already contains per-RBP AUPRC arrays
computed by src/mmpartnet/eval/protocol._eval_held) and produces a
distribution plot for docs/img/per_rbp_distribution.svg.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
data = json.loads((REPO / "mmpartnet_out" / "binding_fair.json").read_text())

# per-RBP AUPRC arrays, one per method + RNA-only baseline
methods = {
    "RNA-only":  [r["rna_multi"] for r in data["methods"]["concat"]["rows"]],   # same array in every row
    "Concat":    [r["real"]      for r in data["methods"]["concat"]["rows"]],
    "FiLM":      [r["real"]      for r in data["methods"]["film"]["rows"]],
    "Cross-att": [r["real"]      for r in data["methods"]["xattn"]["rows"]],
    "Per-res":   [r["real"]      for r in data["methods"]["perres"]["rows"]],
}
labels = list(methods.keys())
values = [methods[k] for k in labels]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# left: violin + strip (density shape + individual RBPs)
positions = np.arange(len(labels))
vp = ax1.violinplot(values, positions=positions, widths=0.75,
                    showmeans=False, showmedians=True, showextrema=False)
for body in vp["bodies"]:
    body.set_alpha(0.4)
    body.set_edgecolor("black")
vp["cmedians"].set_color("black")
for i, v in enumerate(values):
    ax1.scatter(np.random.normal(i, 0.06, size=len(v)), v, s=8, alpha=0.5, color="C0")
ax1.set_xticks(positions); ax1.set_xticklabels(labels)
ax1.set_ylabel("per-RBP AUPRC (K=68)"); ax1.set_title("Distribution — every dot is one RBP")
ax1.grid(axis="y", alpha=0.3)

# right: histogram, overlaid
for lbl, v in methods.items():
    ax2.hist(v, bins=20, alpha=0.4, label=lbl)
ax2.set_xlabel("AUPRC"); ax2.set_ylabel("Number of RBPs"); ax2.set_title("Overlay histogram")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = REPO / "docs" / "img" / "per_rbp_distribution.svg"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")

# also print quick summary numbers for the doc
print("\nsummary:")
for k, v in methods.items():
    v = np.array(v)
    print(f"  {k:10s}  mean={v.mean():.4f}  median={np.median(v):.4f}  "
          f"top-10%={np.quantile(v, 0.9):.4f}  bottom-10%={np.quantile(v, 0.1):.4f}")