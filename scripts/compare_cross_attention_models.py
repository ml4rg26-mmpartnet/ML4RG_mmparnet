#!/usr/bin/env python
"""Benchmark the original vs TFBind cross-attention head over multiple seeds.

This is a thin orchestration wrapper: it calls scripts/train_cross_attention_profile.py
as a subprocess for each (model, seed) so both models see byte-identical data,
sampler, and metrics. It then parses each run's metrics.json and prints a
mean +/- std comparison table.

For each run we report the BEST-over-epochs validation value (matching the
best_pearson / best_auprc checkpoint selection in the trainer):

    profile_pearson   validation profile Pearson (Milestone-2 target)
    binding_auprc     validation binding AUPRC (Milestone-1 target)

Example:
    python scripts/compare_cross_attention_models.py \
        --seeds 0 1 2 --max-train-windows 2000 --max-valid-windows 1000 \
        --tracks 9,138,195 --epochs 5 --batch-size 8 --steps-per-epoch 400 \
        --balanced-train
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "scripts" / "train_cross_attention_profile.py"
DEFAULT_RUN_ROOT = REPO / "mmpartnet_out" / "cross_attention_runs"

MODELS = ["original", "tfbind"]
# (label, path in valid dict, higher_is_better)
METRICS = [
    ("profile_pearson", "pearson", True),
    ("binding_auprc", "binding_auprc", True),
    ("valid_loss", "loss", False),
]


def run_name_for(model: str, mode: str, seed: int) -> str:
    return f"cross_attention_{model}_{mode}_seed{seed}"


def train_one(model: str, seed: int, args: argparse.Namespace) -> Path:
    """Invoke the trainer for one (model, seed); return its run directory."""
    run_dir = Path(args.run_root) / run_name_for(model, args.mode, seed)
    metrics_path = run_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        print(f"[skip] {model} seed{seed}: metrics.json already exists")
        return run_dir

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--model", model,
        "--seed", str(seed),
        "--mode", args.mode,
        "--tracks", args.tracks,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-train-windows", str(args.max_train_windows),
        "--max-valid-windows", str(args.max_valid_windows),
        "--out-dir", str(args.run_root),
    ]
    if args.steps_per_epoch is not None:
        cmd += ["--steps-per-epoch", str(args.steps_per_epoch)]
    if args.balanced_train:
        cmd += ["--balanced-train"]
    cmd += args.passthrough

    print(f"[train] {model} seed{seed}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return run_dir


def best_valid_metrics(run_dir: Path) -> dict[str, float]:
    """Read metrics.json and pick the best-over-epochs validation value per metric."""
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    epochs = metrics.get("epochs", [])
    if not epochs:
        raise ValueError(f"no epochs recorded in {run_dir/'metrics.json'}")

    out: dict[str, float] = {}
    for label, key, higher_better in METRICS:
        values = [
            float(row["valid"][key])
            for row in epochs
            if row.get("valid") is not None and key in row["valid"]
        ]
        if not values:
            continue
        out[label] = max(values) if higher_better else min(values)
    return out


def aggregate(per_seed: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """mean, std across seeds for each metric label."""
    agg: dict[str, tuple[float, float]] = {}
    for label, _key, _hb in METRICS:
        vals = [d[label] for d in per_seed if label in d]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        agg[label] = (mean, std)
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--mode", default="multimodal")
    p.add_argument("--tracks", default="9,138,195")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-train-windows", type=int, default=2000)
    p.add_argument("--max-valid-windows", type=int, default=1000)
    p.add_argument("--steps-per-epoch", type=int, default=400)
    p.add_argument("--balanced-train", action="store_true")
    p.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    p.add_argument("--skip-existing", action="store_true", help="reuse runs whose metrics.json already exists")
    p.add_argument("--summary-out", default=str(DEFAULT_RUN_ROOT / "comparison_summary.json"))
    p.add_argument("passthrough", nargs="*", help="extra flags forwarded verbatim to the trainer")
    args = p.parse_args()

    results: dict[str, dict] = {}
    for model in args.models:
        per_seed = []
        for seed in args.seeds:
            run_dir = train_one(model, seed, args)
            m = best_valid_metrics(run_dir)
            m["seed"] = seed
            per_seed.append(m)
            print(f"  -> {model} seed{seed}: " + ", ".join(
                f"{lbl}={m[lbl]:+.4f}" for lbl, _k, _h in METRICS if lbl in m
            ), flush=True)
        results[model] = {"per_seed": per_seed, "aggregate": aggregate(per_seed)}

    # ---- comparison table ----
    print("\n" + "=" * 68)
    print(f"COMPARISON over seeds {args.seeds}  (best-over-epochs validation)")
    print("=" * 68)
    header = f"{'metric':<18}" + "".join(f"{m:>22}" for m in args.models)
    print(header)
    print("-" * len(header))
    for label, _key, higher_better in METRICS:
        arrow = "up" if higher_better else "down"
        cells = f"{label + ' (' + arrow + ')':<18}"
        for model in args.models:
            agg = results[model]["aggregate"].get(label)
            cells += f"{(f'{agg[0]:+.4f} ± {agg[1]:.4f}' if agg else 'n/a'):>22}"
        print(cells)
    print("=" * 68)

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(
        json.dumps({"config": vars(args), "results": {
            m: {"per_seed": r["per_seed"],
                "aggregate": {k: {"mean": v[0], "std": v[1]} for k, v in r["aggregate"].items()}}
            for m, r in results.items()
        }}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nwrote summary: {args.summary_out}")


if __name__ == "__main__":
    main()
