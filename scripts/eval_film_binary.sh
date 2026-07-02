#!/usr/bin/env bash
# Evaluate the trained FiLM binary-only checkpoint on the validation split.
# Paired with scripts/eval_concat_binary.sh — identical flags except --arch and --checkpoint path.
# Writes eval_metrics.json next to the checkpoint.
set -euo pipefail

# --- environment ----------------------------------------------------------
export ML4RG_REFS=/mnt/storage1/workspace/dgu/parnet_refs
export ML4RG_PARNET_WEIGHTS=/mnt/storage1/ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt
export PYTHONPATH=src

RUN_DIR=/mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask/compare_film_binary_seed0
CHECKPOINT="$RUN_DIR/best_auprc.pt"
OUT="$RUN_DIR/eval_valid_2000_best_auprc.json"

# --- run ------------------------------------------------------------------
pixi run python scripts/eval_film_multitask.py \
  --checkpoint "$CHECKPOINT" \
  --arch film \
  --task binary-only \
  --split valid \
  --max-batches 2000 \
  --hfds /mnt/storage1/ml4rg26-shared/parnet-eclip/data-formatted-for-training/600nt_windows.no-one-hot.stripped/encode.filtered.hfds \
  --protein-h5 /mnt/storage1/ml4rg26-mmparnet/manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5 \
  --track-map mmpartnet_out/prott5_track_map.tsv \
  --binding-dataset /mnt/storage1/ml4rg26-mmparnet/manually_gathered/600nt_windows.no-one-hot.stripped.binding/600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt \
  --mode multimodal \
  --tracks all \
  --include-short \
  --batch-size 32 \
  --device cuda \
  --out "$OUT"

echo
echo "Wrote eval metrics to: $OUT"
