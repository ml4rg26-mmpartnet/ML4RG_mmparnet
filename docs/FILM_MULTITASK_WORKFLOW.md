# FiLM Multitask Workflow

This branch adds a protein+cell FiLM multitask baseline on top of the shared
main-branch project structure.

## Purpose

The model trains on flattened RNA-window/RBP-cell pairs:

```text
RNA window + protein + cell
  -> binding / not-binding prediction
  -> binding profile prediction for positive pairs
```

The RNA encoder is the frozen pretrained PARNET body. The protein condition is
the pooled ProtT5 embedding from the reduced H5 file. Cell line is represented
with a small learnable embedding.

## Main Files

```text
src/mmpartnet/data/multimodal.py       # window-track dataset and collator
src/mmpartnet/models/film.py           # protein+cell FiLM multitask head
src/mmpartnet/models/parnet.py         # frozen PARNET loader
src/mmpartnet/experiments/film_multitask.py  # train/eval loops, metrics, checkpoints
scripts/build_prott5_track_map.py      # build 223-track -> ProtT5 map
scripts/train_film_profile.py          # thin training CLI wrapper
scripts/eval_film_multitask.py         # thin standalone evaluation CLI wrapper
mmpartnet_out/prott5_track_map.tsv     # matched 223-track protein map
```

Training outputs under `mmpartnet_out/film_runs/` are intentionally ignored by
git. Checkpoints should be stored on shared storage, not committed.

## Loss

```text
loss = lambda_profile * profile_loss + lambda_binary * binary_loss
```

Current default experiment uses:

```text
lambda_profile = 1
lambda_binary = 20
profile_mask_source = binding
```

Binary loss is computed for all sampled pairs. Profile loss is computed only for
true positive binding pairs.

## First Formal FiLM Run

```bash
cd /home/dgu/workspace/ML4RG_mmparnet_film

ML4RG_REFS=/home/dgu/workspace/parnet_refs \
ML4RG_PARNET_WEIGHTS=/home/dgu/storage_ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt \
PYTHONPATH=src \
/home/dgu/venvs/torch39/bin/python scripts/train_film_profile.py \
  --mode multimodal \
  --tracks all \
  --binding-dataset /home/dgu/storage_ml4rg26-mmparnet/manually_gathered/600nt_windows.no-one-hot.stripped.binding/600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt \
  --max-train-windows 65536 \
  --max-valid-windows 16384 \
  --batch-size 32 \
  --epochs 5 \
  --balanced-train \
  --balanced-pos-fraction 0.5 \
  --steps-per-epoch 1000 \
  --max-valid-batches 1000 \
  --lambda-binary 20 \
  --device cuda \
  --include-short \
  --progress-every 250 \
  --run-name formal_pureclip_l20_5x1000_seed0
```

Resume from the latest full checkpoint:

```bash
--resume mmpartnet_out/film_runs/formal_pureclip_l20_5x1000_seed0/last.pt
```

## Checkpoints

The training script writes:

```text
best.pt            # compatibility name, same as Pearson-best
best_pearson.pt    # best validation profile Pearson
best_auprc.pt      # best validation binding AUPRC
last.pt            # latest full checkpoint, resumable
last.statedict.pt  # latest model weights only
metrics.json       # training/validation history
```

## Standalone Validation

Use validation for model selection. Do not use test until the final model and
hyperparameters are fixed.

```bash
ML4RG_REFS=/home/dgu/workspace/parnet_refs \
ML4RG_PARNET_WEIGHTS=/home/dgu/storage_ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt \
PYTHONPATH=src \
/home/dgu/venvs/torch39/bin/python scripts/eval_film_multitask.py \
  --checkpoint mmpartnet_out/film_runs/formal_pureclip_l20_5x1000_seed0/best_pearson.pt \
  --split valid \
  --max-batches 2000 \
  --device cuda \
  --include-short \
  --out mmpartnet_out/film_runs/formal_pureclip_l20_5x1000_seed0/eval_valid_2000_best_pearson.json
```

## Current FiLM Baseline Result

On the `formal_pureclip_l20_5x1000_seed0` run, after 15k training steps:

```text
best validation Pearson: epoch 13, 0.4812
best validation AUPRC:   epoch 10, 0.2377
```

Using the same valid-2000 evaluation for checkpoint comparison:

```text
best_pearson.pt: Pearson 0.4753, AUPRC 0.2107
best_auprc.pt:   Pearson 0.4703, AUPRC 0.2106
last.pt:         Pearson 0.4705, AUPRC 0.2171
```

For a single FiLM baseline checkpoint, use `best_pearson.pt` because profile
prediction is the primary signal-profile objective and its AUPRC is similar to
the other checkpoints.

## Single-Task Ablations

The same workflow can train single-task controls.

Binary-only control:

```text
lambda_profile = 0
lambda_binary = 1
balanced_pos_fraction = 0.5
steps_per_epoch = 1000
```

Profile-only control:

```text
lambda_profile = 1
lambda_binary = 0
balanced_pos_fraction = 1.0
steps_per_epoch = 500
```

The profile-only setup uses positive-only sampling because negative pairs do not
contribute profile loss. With batch size 32, this gives about 16k positive
profile examples per epoch, matching the multitask setup's 50/50 sampling with
1000 steps per epoch.
