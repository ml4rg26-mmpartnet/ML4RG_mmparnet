#!/usr/bin/env python
"""Benchmark cross-attention configurations over multiple seeds.

Thin orchestration wrapper: it calls scripts/train_cross_attention_profile.py as
a subprocess for each (config, seed) so every configuration sees byte-identical
data, sampler, and metrics. It then parses each run's metrics.json and prints a
mean +/- std comparison table.

A "config" is ``model:num_blocks`` (e.g. ``tfbind:3``). This lets you compare
architectures at a fixed depth AND test whether depth (our asymmetric
bidirectional stacking) helps, in one run:

    --configs original:1 tfbind:1 tfbind:3

- original:1 vs tfbind:1  -> architecture at equal depth (fusion internals)
- tfbind:1 vs tfbind:3    -> does stacking blocks (our idea) help?

For each run we report the BEST-over-epochs validation value:

    profile_pearson   validation profile Pearson (Milestone-2 target)
    binding_auprc     validation binding AUPRC (meaningless if --lambda-binary 0)

Example (profile-only, lr 3e-4, 15k windows, all tracks, 3 seeds):
    python scripts/compare_cross_attention_models.py \
        --configs original:1 tfbind:1 tfbind:3 \
        --seeds 0 1 2 --tracks all \
        --max-train-windows 15000 --max-valid-windows 5000 \
        --epochs 15 --batch-size 32 --steps-per-epoch 500 --balanced-train \
        --run-tag prof_lr3e4_15kw \
        -- \
        --lr 3e-4 --lambda-binary 0 --balanced-pos-fraction 1.0 \
        --include-short --max-valid-batches 1000 --profile-mask-source binding \
        --max-protein-len 1024
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

# (label, path in valid dict, higher_is_better)
METRICS = [
    ("profile_pearson", "pearson", True),
    ("binding_auprc", "binding_auprc", True),
    ("valid_loss", "loss", False),
]


def parse_config(spec: str) -> tuple[str, int]:
    """'tfbind:3' -> ('tfbind', 3); 'original' -> ('original', 1)."""
    if ":" in spec:
        model, nb = spec.split(":", 1)
        return model, int(nb)
    return spec, 1


def config_label(model: str, num_blocks: int) -> str:
    return f"{model}_nb{num_blocks}"


def run_name_for(model: str, num_blocks: int, mode: str, seed: int, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"cross_attention_{config_label(model, num_blocks)}_{mode}{suffix}_seed{seed}"


def train_one(model: str, num_blocks: int, seed: int, args: argparse.Namespace) -> Path:
    """Invoke the trainer for one (config, seed); return its run directory."""
    run_dir = Path(args.run_root) / run_name_for(model, num_blocks, args.mode, seed, args.run_tag)
    if args.skip_existing and (run_dir / "metrics.json").exists():
        print(f"[skip] {config_label(model, num_blocks)} seed{seed}: metrics.json exists")
        return run_dir

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--model", model,
        "--num-blocks", str(num_blocks),
        "--seed", str(seed),
        "--mode", args.mode,
        "--tracks", args.tracks,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-train-windows", str(args.max_train_windows),
        "--max-valid-windows", str(args.max_valid_windows),
        "--out-dir", str(args.run_root),
        "--run-name", run_dir.name,
    ]
    if args.steps_per_epoch is not None:
        cmd += ["--steps-per-epoch", str(args.steps_per_epoch)]
    if args.balanced_train:
        cmd += ["--balanced-train"]
    cmd += args.passthrough

    print(f"[train] {config_label(model, num_blocks)} seed{seed}: {' '.join(cmd)}", flush=True)
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
    p.add_argument("--configs", nargs="+", default=["original:1", "tfbind:1", "tfbind:3"],
                   help="configurations as model:num_blocks (e.g. tfbind:3)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--mode", default="multimodal")
    p.add_argument("--tracks", default="9,138,195")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-train-windows", type=int, default=15000)
    p.add_argument("--max-valid-windows", type=int, default=5000)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--balanced-train", action="store_true")
    p.add_argument("--run-tag", default="", help="suffix in run dir names to keep configs distinct")
    p.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    p.add_argument("--skip-existing", action="store_true", help="reuse runs whose metrics.json already exists")
    p.add_argument("--summary-out", default=str(DEFAULT_RUN_ROOT / "comparison_summary.json"))
    p.add_argument("passthrough", nargs="*", help="extra flags forwarded verbatim to the trainer (after --)")
    args = p.parse_args()

    configs = [parse_config(c) for c in args.configs]
    labels = [config_label(m, nb) for m, nb in configs]

    results: dict[str, dict] = {}
    for (model, num_blocks), label in zip(configs, labels):
        per_seed = []
        for seed in args.seeds:
            run_dir = train_one(model, num_blocks, seed, args)
            m = best_valid_metrics(run_dir)
            m["seed"] = seed
            per_seed.append(m)
            print(f"  -> {label} seed{seed}: " + ", ".join(
                f"{lbl}={m[lbl]:+.4f}" for lbl, _k, _h in METRICS if lbl in m
            ), flush=True)
        results[label] = {"per_seed": per_seed, "aggregate": aggregate(per_seed)}

    # ---- comparison table ----
    print("\n" + "=" * (20 + 22 * len(labels)))
    print(f"COMPARISON over seeds {args.seeds}  (best-over-epochs validation)")
    print("=" * (20 + 22 * len(labels)))
    header = f"{'metric':<20}" + "".join(f"{lbl:>22}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for label, _key, higher_better in METRICS:
        arrow = "up" if higher_better else "down"
        cells = f"{label + ' (' + arrow + ')':<20}"
        for cfg_label in labels:
            agg = results[cfg_label]["aggregate"].get(label)
            cells += f"{(f'{agg[0]:+.4f} ± {agg[1]:.4f}' if agg else 'n/a'):>22}"
        print(cells)
    print("=" * (20 + 22 * len(labels)))

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(
        json.dumps({"config": vars(args), "results": {
            lbl: {"per_seed": r["per_seed"],
                  "aggregate": {k: {"mean": v[0], "std": v[1]} for k, v in r["aggregate"].items()}}
            for lbl, r in results.items()
        }}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nwrote summary: {args.summary_out}")


if __name__ == "__main__":
    main()
