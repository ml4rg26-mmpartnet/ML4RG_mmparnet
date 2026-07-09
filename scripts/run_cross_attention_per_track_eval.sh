#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-/home/dgu/workspace/cross_attention_runs/per_track_test}"
WINDOW_SEED="${WINDOW_SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
export ML4RG_REFS="${ML4RG_REFS:-/home/dgu/workspace/parnet_refs}"
export ML4RG_PARNET_WEIGHTS="${ML4RG_PARNET_WEIGHTS:-/home/dgu/storage_ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt}"

RUNS=/home/dgu/workspace/cross_attention_runs
PROFILE="$RUNS/profile_only_lr3e4_latent256_15x500_seed0/best_pearson.pt"
BINARY="$RUNS/binary_only_l10_lr3e4_latent256_15x1000_seed0/best_auprc.pt"
MULTI_PEARSON="$RUNS/multitask_l10_lr3e4_latent256_5x1000_seed0/best_pearson.pt"

mkdir -p "$OUT_DIR"

run_eval() {
  local name=$1 task=$2 checkpoint=$3 panel=$4 selection=$5
  "$PYTHON" scripts/eval_cross_attention_per_track.py \
    --checkpoint "$checkpoint" \
    --task "$task" \
    --panel "$panel" \
    --window-selection "$selection" \
    --window-count 15000 \
    --window-seed "$WINDOW_SEED" \
    --batch-size 32 \
    --device "$DEVICE" \
    --out "$OUT_DIR/$name.json" \
    2>&1 | tee "$OUT_DIR/$name.log"
}

# Common comparison panel: first 15k length-600 test windows and 68 shared tracks.
run_eval common68_profile_only profile-only "$PROFILE" common68 reference-first
run_eval common68_binary_only binary-only "$BINARY" common68 reference-first
run_eval common68_multitask multitask "$MULTI_PEARSON" common68 reference-first

# Expanded panel: one fixed random 15k-window subset shared by every checkpoint.
run_eval alltracks_profile_only profile-only "$PROFILE" all random
run_eval alltracks_binary_only binary-only "$BINARY" all random
run_eval alltracks_multitask multitask "$MULTI_PEARSON" all random

"$PYTHON" scripts/plot_cross_attention_per_track.py \
  "$OUT_DIR/common68_profile_only.json" \
  "$OUT_DIR/common68_binary_only.json" \
  "$OUT_DIR/common68_multitask.json" \
  --out "$OUT_DIR/common68_task_distributions.png"

"$PYTHON" scripts/plot_cross_attention_per_track.py \
  "$OUT_DIR/alltracks_profile_only.json" \
  "$OUT_DIR/alltracks_binary_only.json" \
  "$OUT_DIR/alltracks_multitask.json" \
  --out "$OUT_DIR/alltracks_task_distributions.png"
