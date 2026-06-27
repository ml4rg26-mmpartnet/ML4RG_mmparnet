# FiLM Multitask Baseline

## Dataset Background

The original PARNET-style dataset is organized around fixed-length RNA windows.
For each RNA window, the dataset contains binding signal tracks for many RBP-cell
experiments. In our current setup there are 223 tracks. A track is not just a
protein name; it is a protein measured in a specific cell line, for example:

```text
QKI_HepG2
QKI_K562
U2AF2_HepG2
```

This matters because the same RBP can have different observed binding behavior
in different cell lines. Therefore, protein identity alone is not enough to
fully define the prediction target. The model should know both:

```text
which RBP protein are we asking about?
which cell line was the experiment measured in?
```

The original PARNET model predicts all tracks at once from RNA sequence alone:

```text
RNA window -> 223 binding signal profiles
```

This branch reformulates the problem as a conditional multimodal task. Instead
of predicting all 223 tracks at once, each training example asks about one RNA
window under one RBP-cell condition:

```text
RNA window + RBP protein + cell line -> prediction for that RBP-cell track
```

## Baseline Overview

This branch implements a protein + cell conditioned FiLM baseline. The RNA side
uses the frozen pretrained PARNET body as an RNA feature extractor. The protein
side uses a pooled ProtT5 embedding. The cell line is represented by a small
learnable embedding.

The model predicts two things:

```text
1. binding / not-binding
2. binding signal profile across the RNA window
```

The high-level data flow is:

```text
RNA window
  -> frozen PARNET RNA encoder
  -> RNA feature

RBP + cell line
  -> condition embedding

RNA feature + condition embedding
  -> FiLM conditioning
  -> binary binding head
  -> profile prediction head
```

The rest of this document explains what FiLM is, why the branch uses multitask
learning, how the data flows through the code, and what the current validation
results look like.

## Why FiLM(Feature-wise Linear Modulation)

It is a simple way to let one input condition another input. In this project,
the main input with position-level information is the RNA feature from PARNET:

```text
RNA feature: [channels, 600 positions]
```

The protein and cell line are conditions. They do not have a 600-position axis;
they describe which RBP/cell context we are asking about:

```text
condition = protein embedding + cell embedding
```

A very simple fusion method would be to concatenate the condition to the RNA
feature. FiLM does something more targeted: it uses the condition to produce two
sets of channel-wise parameters:

```text
raw_gamma, beta = MLP(condition)
```

Then these parameters modulate the RNA feature channels:

```text
scale = 1 + tanh(raw_gamma)
conditioned RNA feature = scale * RNA feature + beta
```

Here, `scale` decides which RNA feature channels should be amplified or reduced,
and `beta` shifts the channel values. The same scale and beta are broadcast
across all RNA positions, so every position is interpreted under the same
protein/cell condition.

This is useful because different RBPs may care about different RNA patterns. For
example, one RBP-cell condition can make the model pay more attention to one set
of PARNET channels, while another condition can emphasize a different set of
channels.

This branch uses `1 + tanh(raw_gamma)` instead of directly using `raw_gamma` as
the scale. This makes the FiLM layer start closer to the original frozen PARNET
feature: if `raw_gamma` is near zero, then `scale` is near one. In other words,
the model begins close to "use the pretrained RNA feature as is" and then learns
how each protein/cell condition should adjust it.

## Why Multitask

The original PARNET model is mainly a profile prediction model. For each RNA
window and track, it predicts a normalized binding profile across the 600 nt RNA
window. 

During profile training, not every window-track pair contributes to the loss.
The training code first sums the observed eCLIP reads across the whole RNA
window for that track. If the total read count is below a threshold, for example
`min_count = 10`, the model still runs the forward pass and still produces a
predicted profile, but this pair is filtered out of the profile loss:

```text
predicted profile is computed
true eCLIP reads across the whole window < min_count
  -> profile loss is not computed for this pair
  -> this pair does not contribute to backpropagation
  -> this pair does not update the model parameters through profile loss
```

This is reasonable for profile prediction because a very low-read window may not
have a reliable profile shape. However, it also means the model mostly learns
from window-track pairs with enough reads. In other words, the profile objective
teaches the model what signal-containing examples look like, but it gives much
weaker supervision about what no-signal or non-binding examples look like.

This also matters at inference time. The profile head will always output a
normalized profile, even for an RNA-protein-cell pair that truly has no binding
signal. In that case, the predicted profile is hard to interpret: the model is
still forced to distribute probability mass across positions, even though there
may be no real binding event to localize.

To address this, this branch adds a binary binding task in addition to the
profile task:

```text
binary head:
  RNA + protein + cell -> binding probability

profile head:
  RNA + protein + cell -> target/control/total signal profile
```

During training and validation, the profile objective is gated by the true
binding label, not by the model's predicted binding probability:

```text
if true binding label = 1:
  compute profile loss / profile Pearson for this pair

if true binding label = 0:
  compute binary loss only, and do not use this pair for profile loss
```

At inference time, the true binding label is not available. The binary head can
then be used as an auxiliary confidence score for whether the predicted profile
should be interpreted as a real binding profile.

In `--task multitask` mode, both heads are active and the total loss is:

```text
loss = lambda_profile * profile_loss + lambda_binary * binary_loss
```

In all current multitask experiments, we fix `lambda_profile = 1` and tune only
`lambda_binary` to control how strongly the binary task affects training:

```text
lambda_profile = 1
lambda_binary = 10, 15, or 20
```

In multitask mode, binary loss is computed for sampled positive and negative
pairs. For profile loss, the current experiments use the true binding label as
the mask: profile loss is computed only for positive binding examples, because
profile shape is only meaningful when there is binding signal to explain. The
training code exposes this as `--profile-mask-source binding`, but this is the
default setting used in this branch.

The single-task ablations are implemented with the `--task` argument:

```text
--task binary-only
  -> forward only the binary head
  -> do not compute target/control/mix/profile outputs

--task profile-only
  -> forward only the profile head
  -> do not compute the binary binding head
```

This avoids wasting compute on the unused output head.

## Data Flow

```text
RNA sequence
  -> one-hot RNA [4, 600]
  -> frozen PARNET body
  -> RNA feature [D, 600]

RBP-cell track name, for example QKI_HepG2
  -> protein name QKI
  -> pooled ProtT5 protein embedding through mmpartnet.protein
  -> cell line HepG2
  -> learnable cell embedding

protein embedding + cell embedding
  -> FiLM gamma/beta
  -> conditioned RNA feature

conditioned RNA feature
  -> binary head
  -> binding probability

conditioned RNA feature
  -> profile head
  -> target profile + control profile + mix coefficient
  -> predicted eCLIP profile
```

Windows shorter than 600 nt are included when `--include-short` is used. They
are padded to length 600, and a mask is used so padding positions do not
contribute to pooling, profile softmax, or profile loss.

## Main Files

```text
src/mmpartnet/data/multimodal.py
  Builds the FiLM-specific flattened RNA-window/RBP-cell examples and batches
  them for training.

src/mmpartnet/protein/providers/prott5_h5.py
  Registers pooled ProtT5 H5 embeddings under the repository's swappable
  mmpartnet.protein interface.

src/mmpartnet/models/film.py
  Defines the protein+cell FiLM model and its multitask, binary-only, and
  profile-only output modes.

src/mmpartnet/models/parnet.py
  Loads the frozen pretrained PARNET body.

src/mmpartnet/experiments/film_multitask.py
  Contains training, validation, metrics, balanced sampling, and checkpoint
  logic.

scripts/build_prott5_track_map.py
  Builds the mapping from 223 PARNET tracks to ProtT5 protein embeddings.

scripts/train_film_profile.py
  Thin CLI wrapper for training.

scripts/eval_film_multitask.py
  Thin CLI wrapper for standalone validation/evaluation.

mmpartnet_out/prott5_track_map.tsv
  Matched RBP-cell track to protein embedding map.
```

Checkpoints and logs are experiment artifacts and are not committed to GitHub.
Following the VM storage guideline, the current reusable FiLM checkpoints and
validation results are stored in the group results directory:

```text
/mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask
```

This directory contains the selected checkpoints, `metrics.json`, validation
JSON files, and training logs for the formal runs listed below. The same files
also remain in the original development checkout under
`$HOME/workspace/ML4RG_mmparnet_film/mmpartnet_out/film_runs/`, but the group
results directory above is the path other people should use.

## Current Results

All results below use the validation split, not the test split. The test split
has not been used yet and should only be used after the final model design,
loss weights, and checkpoint selection rule are fixed.

`train-time best` means the best validation metric observed during training.
`valid-2000` means the standalone evaluator was run on up to 2000 validation
batches using the selected checkpoint.

| model | epochs | selected checkpoint | train-time best Pearson | train-time best AUPRC | valid-2000 Pearson | valid-2000 AUPRC |
|---|---:|---|---:|---:|---:|---:|
| multitask, `lambda_binary=10` | 15 | `best_pearson.pt` | 0.4888 | 0.2259 | 0.4829 | 0.2131 |
| multitask, `lambda_binary=20` | 15 | `best_pearson.pt` | 0.4812 | 0.2377 | 0.4753 | 0.2107 |
| profile-only | 15 | `best_pearson.pt` | 0.4940 | N/A | 0.4875 | N/A |
| binary-only | 15 | `best_auprc.pt` | N/A | 0.2303 | N/A | 0.2020 |

Direct checkpoint paths:

```text
multitask, lambda_binary=10:
  /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask/formal_pureclip_l10_10x1000_seed0/best_pearson.pt

multitask, lambda_binary=20:
  /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask/formal_pureclip_l20_5x1000_seed0/best_pearson.pt

profile-only:
  /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask/formal_pureclip_profile_only_15x500_seed0/best_pearson.pt

binary-only:
  /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask/formal_pureclip_binary_only_15x1000_seed0/best_auprc.pt
```

Current observation:

```text
profile-only gives the best profile Pearson.
binary-only gives a similar or slightly lower binding AUPRC than multitask.
multitask does not clearly beat the single-task baselines in this run.
Among multitask runs, lambda_binary=10 is better than lambda_binary=20 for profile Pearson.
```

One likely reason is that the profile task and binary task are not asking for
exactly the same type of information. Profile prediction needs fine
position-level signal: among the 600 RNA positions, where should the binding
signal concentrate? Binary prediction is more global: does this RNA-window/RBP-
cell pair have binding signal at all?

When both losses update the same FiLM-conditioned RNA representation, the binary
task can push the shared representation toward global binding/non-binding
features. That can help classification, but it may make the representation less
specialized for the detailed profile shape. This is consistent with the current
results: profile-only has the best Pearson, while multitask does not improve the
profile objective.

Increasing `lambda_binary` from 10 to 20 also did not help the profile metric.
This suggests that putting too much weight on binary supervision may interfere
with the position-level profile objective. A possible next experiment is to let
binary loss update only the binary head, while profile loss updates the shared
FiLM/profile pathway.

## How To Run

### Multitask

This trains both heads:

```text
loss = profile_loss + lambda_binary * binary_loss
```

Use `lambda_binary=10` for the current best multitask profile setting:

```bash
cd $HOME/workspace/ML4RG_mmparnet
# If your checkout has a different folder name, cd into that folder instead.

# Activate a Python environment with this repo's requirements installed first.
# Example: source /path/to/your/env/bin/activate

mkdir -p /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask

ML4RG_REFS=/mnt/storage1/workspace/dgu/parnet_refs \
ML4RG_PARNET_WEIGHTS=/mnt/storage1/ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt \
PYTHONPATH=src \
python scripts/train_film_profile.py \
  --task multitask \
  --mode multimodal \
  --tracks all \
  --binding-dataset /mnt/storage1/ml4rg26-mmparnet/manually_gathered/600nt_windows.no-one-hot.stripped.binding/600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt \
  --max-train-windows 65536 \
  --max-valid-windows 16384 \
  --batch-size 32 \
  --epochs 15 \
  --balanced-train \
  --balanced-pos-fraction 0.5 \
  --steps-per-epoch 1000 \
  --max-valid-batches 1000 \
  --lambda-profile 1 \
  --lambda-binary 10 \
  --device cuda \
  --include-short \
  --progress-every 250 \
  --out-dir /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask \
  --run-name formal_pureclip_l10_10x1000_seed0
```

To run the `lambda_binary=20` version, change:

```text
--lambda-binary 20
--run-name formal_pureclip_l20_15x1000_seed0
```

### Binary-only

This trains only the binding / not-binding head. The profile head is not
forwarded.

```bash
cd $HOME/workspace/ML4RG_mmparnet
# If your checkout has a different folder name, cd into that folder instead.

mkdir -p /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask

ML4RG_REFS=/mnt/storage1/workspace/dgu/parnet_refs \
ML4RG_PARNET_WEIGHTS=/mnt/storage1/ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt \
PYTHONPATH=src \
python scripts/train_film_profile.py \
  --task binary-only \
  --mode multimodal \
  --tracks all \
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
  --out-dir /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask \
  --run-name formal_pureclip_binary_only_15x1000_seed0
```

### Profile-only

This trains only the signal-profile head. The binary head is not forwarded.
Positive-only sampling is used because negative examples do not contribute
profile loss.

```bash
cd $HOME/workspace/ML4RG_mmparnet
# If your checkout has a different folder name, cd into that folder instead.

mkdir -p /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask

ML4RG_REFS=/mnt/storage1/workspace/dgu/parnet_refs \
ML4RG_PARNET_WEIGHTS=/mnt/storage1/ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt \
PYTHONPATH=src \
python scripts/train_film_profile.py \
  --task profile-only \
  --mode multimodal \
  --tracks all \
  --binding-dataset /mnt/storage1/ml4rg26-mmparnet/manually_gathered/600nt_windows.no-one-hot.stripped.binding/600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt \
  --max-train-windows 65536 \
  --max-valid-windows 16384 \
  --batch-size 32 \
  --epochs 15 \
  --balanced-train \
  --balanced-pos-fraction 1.0 \
  --steps-per-epoch 500 \
  --max-valid-batches 1000 \
  --device cuda \
  --include-short \
  --progress-every 250 \
  --out-dir /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask \
  --run-name formal_pureclip_profile_only_15x500_seed0
```

### Resume Training

Resume from the latest full checkpoint by adding:

```bash
--resume /mnt/storage1/ml4rg26-mmparnet/ML4RG_mmparnet/results/film_multitask/<run-name>/last.pt
```

Each training run writes:

```text
best.pt            # compatibility name, same as Pearson-best
best_pearson.pt    # best validation profile Pearson
best_auprc.pt      # best validation binding AUPRC
last.pt            # latest full checkpoint, resumable
last.statedict.pt  # latest model weights only
metrics.json       # training/validation history
```
