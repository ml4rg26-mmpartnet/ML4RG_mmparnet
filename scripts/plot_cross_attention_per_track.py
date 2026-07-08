#!/usr/bin/env python
"""Plot per-RBP-cell Pearson and AUPRC distributions from evaluation JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    labels = [record["task"] for record in records]
    metric_data = {
        "Pearson": [
            [row["pearson"] for row in record["rows"] if row["pearson"] is not None]
            for record in records
        ],
        "AUPRC": [
            [row["auprc"] for row in record["rows"] if row["auprc"] is not None]
            for record in records
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    colors = ["#3B82F6", "#E4572E", "#2A9D8F"]
    rng = np.random.default_rng(0)
    for ax, (metric, groups) in zip(axes, metric_data.items()):
        present = [(i, values) for i, values in enumerate(groups) if values]
        positions = [i + 1 for i, _ in present]
        values = [values for _, values in present]
        if values:
            violin = ax.violinplot(values, positions=positions, showextrema=False)
            for body, (i, _) in zip(violin["bodies"], present):
                body.set_facecolor(colors[i % len(colors)])
                body.set_alpha(0.35)
            ax.boxplot(values, positions=positions, widths=0.22, showfliers=False)
            for position, (i, group) in zip(positions, present):
                jitter = rng.normal(position, 0.035, len(group))
                ax.scatter(jitter, group, s=12, alpha=0.55, color=colors[i % len(colors)])
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=18, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Per RBP-cell {metric}")
        ax.grid(axis="y", alpha=0.25)

    panel = records[0]["panel"]
    n_windows = records[0]["n_windows"]
    fig.suptitle(f"Cross-attention task comparison: {panel}, {n_windows:,} test windows")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
