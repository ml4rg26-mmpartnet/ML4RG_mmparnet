#!/usr/bin/env bash
# Early-fusion concat-head binary-only baseline.
# Paired with scripts/train_film_binary.sh — identical flags except --arch.
# Concat pools the RNA dimension away, so this head only supports the binary task.
set -euo pipefail

# --- environment ----------------------------------------------------------
export ML4RG_REFS=/mnt/storage1/workspace/dgu/parnet_refs
export ML4RG_PARNET_WEIGHTS=/mnt/storage1/ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt
export PYTHONPATH=src

OUT_DIR=/mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask
RUN_NAME=compare_concat_binary_seed0
mkdir -p "$OUT_DIR"

# --- run ------------------------------------------------------------------
pixi run python scripts/train_film_profile.py \
  --task binary-only \
  --arch concat \
  --mode multimodal \
  --tracks all \
  --hfds /mnt/storage1/ml4rg26-shared/parnet-eclip/data-formatted-for-training/600nt_windows.no-one-hot.stripped/encode.filtered.hfds \
  --protein-h5 /mnt/storage1/ml4rg26-mmparnet/manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5 \
  --track-map mmpartnet_out/prott5_track_map.tsv \
  --binding-dataset /mnt/storage1/ml4rg26-mmparnet/manually_gathered/600nt_windows.no-one-hot.stripped.binding/600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt \
  --max-train-windows 65536 \
  --max-valid-windows 16384 \
  --batch-size 32 \
  --epochs 15 \
  --balanced-train \
  --balanced-pos-fraction 0.5 \
  --steps-per-epoch 1000 \
  --max-valid-batches 1000 \
  --device cuda \
  --include-short \
  --progress-every 250 \
  --seed 0 \
  --out-dir "$OUT_DIR" \
  --run-name "$RUN_NAME"
