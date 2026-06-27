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

Current multitask experiments keep the profile loss as the reference scale and
mainly tune the binary loss weight:

```text
lambda_profile = 1
lambda_binary = 10, 15, or 20
profile_mask_source = binding
```

`lambda_profile` is still exposed in the code for flexibility, but all current
experiments fix it to 1 and compare different `lambda_binary` values.

The single-task ablations are implemented with the `--task` argument, not only
by setting one loss weight to zero:

```text
--task binary-only
  -> forward only the binary head
  -> do not compute target/control/mix/profile outputs

--task profile-only
  -> forward only the profile head
  -> do not compute the binary binding head
```

This avoids wasting compute on the unused output head. In multitask mode, binary
loss is computed for sampled positive and negative pairs. Profile loss is
computed only for true positive binding pairs, because profile shape is only
meaningful when there is binding signal to explain.

## Data Flow

```text
RNA sequence
  -> one-hot RNA [4, 600]
  -> frozen PARNET body
  -> RNA feature [D, 600]

RBP-cell track name, for example QKI_HepG2
  -> protein name QKI
  -> pooled ProtT5 protein embedding
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
  -> total/eCLIP profile
```

Windows shorter than 600 nt are included when `--include-short` is used. They
are padded to length 600, and a mask is used so padding positions do not
contribute to pooling, profile softmax, or profile loss.

## Main Files

```text
src/mmpartnet/data/multimodal.py
  Builds RNA-window/RBP-cell examples and batches them for training.

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

Training outputs are written under:

```text
mmpartnet_out/film_runs/<run-name>/
```

Checkpoints and logs are experiment artifacts and are not committed to GitHub.
They should stay in the shared VM/storage output directory for the run. The
exact shared checkpoint location can be filled in once the current experiments
are finished and the recommended checkpoint is selected.

## Current Results

All results below use the validation split, not the test split. Test should only
be used after the final model and hyperparameters are fixed.

### Multitask Baseline, Lambda 20

Run:

```text
formal_pureclip_l20_5x1000_seed0
```

This run was trained for 15k total steps. During training:

```text
best validation Pearson: epoch 13, 0.4812
best validation AUPRC:   epoch 10, 0.2377
```

Using standalone valid-2000 evaluation:

```text
best_pearson.pt: Pearson 0.4753, AUPRC 0.2107
best_auprc.pt:   Pearson 0.4703, AUPRC 0.2106
last.pt:         Pearson 0.4705, AUPRC 0.2171
```

For the first multitask baseline checkpoint, `best_pearson.pt` is the preferred
choice because profile prediction is the primary signal-profile objective.

### Single-Task Ablations

Two single-task controls were trained to check whether multitask learning is
actually helping.

```text
binary-only:
  task = binary-only
  lambda_profile = 0
  lambda_binary = 1
  balanced_pos_fraction = 0.5
  steps_per_epoch = 1000
  epochs = 15

profile-only:
  task = profile-only
  lambda_profile = 1
  lambda_binary = 0
  balanced_pos_fraction = 1.0
  steps_per_epoch = 500
  epochs = 15
```

The profile-only setup uses positive-only sampling because negative pairs do not
contribute profile loss. With batch size 32, 500 steps gives about 16k positive
profile examples per epoch, matching the multitask setup's 50/50 sampling with
1000 steps per epoch.

Standalone valid-2000 results:

```text
multitask best_pearson.pt:    Pearson 0.4753, AUPRC 0.2107
binary-only best_auprc.pt:    AUPRC 0.2020
profile-only best_pearson.pt: Pearson 0.4875
```

Interpretation:

```text
profile-only > multitask for profile Pearson
multitask > binary-only for binding AUPRC
```

This suggests that the binary task can help binding classification, but with
`lambda_binary = 20` it may slightly hurt the position-level profile objective.
This is plausible because binary classification learns more global
binding/non-binding features, while profile prediction needs fine position-level
signal information.

## Overall Validation Comparison

The table below compares the main ablations and lambda sweep runs. `train-time`
metrics are the best validation metrics observed during training. `valid-2000`
metrics use the standalone evaluator with the same validation setting across
runs.

| model | epochs | train-time best Pearson | train-time best AUPRC | valid-2000 Pearson | valid-2000 AUPRC |
|---|---:|---:|---:|---:|---:|
| profile-only | 15 | 0.4940 | N/A | 0.4875 | N/A |
| binary-only | 15 | N/A | 0.2303 | N/A | 0.2020 |
| multitask lambda=10 | 15 | 0.4888 | 0.2259 | 0.4829 | 0.2131 |
| multitask lambda=15 | 10 | 0.4819 | 0.2224 | 0.4742 | 0.2080 |
| multitask lambda=20 | 15 | 0.4812 | 0.2377 | 0.4753 | 0.2107 |

Summary:

```text
Profile-only gives the best profile Pearson overall.
Among multitask models, lambda=10 gives the best profile Pearson.
Binary-only does not outperform multitask on valid-2000 AUPRC.
lambda=20 had the best training-time AUPRC, but lower profile Pearson.
```

This suggests that binary supervision can help the binding classifier, but too
large a binary loss weight can interfere with the position-level profile task.
For the current FiLM multitask profile baseline, `lambda_binary = 10` is the best
validation-selected setting.

## Lambda Sweep Results

Because `lambda_binary = 20` may put too much weight on the binary task, two
additional multitask runs were trained with smaller binary loss weights:

```text
lambda_binary = 10
lambda_binary = 15
```

Both runs used:

```text
task = multitask
lambda_profile = 1
tracks = all
binding dataset = pureCLIP
balanced_pos_fraction = 0.5
steps_per_epoch = 1000
include_short = true
```

Training-time best validation metrics:

```text
lambda=10, 15 epochs:
  best Pearson: epoch 13, 0.4888, AUPRC 0.2183
  best AUPRC:   epoch 9,  0.2259, Pearson 0.4847

lambda=15, 10 epochs:
  best Pearson: epoch 9,  0.4819, AUPRC 0.2224
  best AUPRC:   epoch 9,  0.2224, Pearson 0.4819

lambda=20, 15 epochs:
  best Pearson: epoch 13, 0.4812, AUPRC 0.2156
  best AUPRC:   epoch 10, 0.2377, Pearson 0.4776
```

Standalone valid-2000 evaluation for the best Pearson checkpoint:

```text
lambda=10 best_pearson.pt: Pearson 0.4829, AUPRC 0.2131
lambda=15 best_pearson.pt: Pearson 0.4742, AUPRC 0.2080
lambda=20 best_pearson.pt: Pearson 0.4753, AUPRC 0.2107
```

Current interpretation:

```text
lambda=10 gives the best multitask profile Pearson.
lambda=20 gives the best training-time AUPRC, but its profile Pearson is lower.
lambda=15 does not improve over lambda=10 or lambda=20.
```

For the current FiLM multitask profile baseline, use:

```text
formal_pureclip_l10_10x1000_seed0/best_pearson.pt
```

This checkpoint came from the resumed 15-epoch `lambda_binary = 10` run. It is
not a final test-set result; it was selected using validation metrics.

## How To Train

Example multitask training command:

```bash
cd /home/dgu/workspace/ML4RG_mmparnet_film

ML4RG_REFS=/home/dgu/workspace/parnet_refs \
ML4RG_PARNET_WEIGHTS=/home/dgu/storage_ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt \
PYTHONPATH=src \
/home/dgu/venvs/torch39/bin/python scripts/train_film_profile.py \
  --task multitask \
  --mode multimodal \
  --tracks all \
  --binding-dataset /home/dgu/storage_ml4rg26-mmparnet/manually_gathered/600nt_windows.no-one-hot.stripped.binding/600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt \
  --max-train-windows 65536 \
  --max-valid-windows 16384 \
  --batch-size 32 \
  --epochs 10 \
  --balanced-train \
  --balanced-pos-fraction 0.5 \
  --steps-per-epoch 1000 \
  --max-valid-batches 1000 \
  --lambda-profile 1 \
  --lambda-binary 10 \
  --device cuda \
  --include-short \
  --progress-every 250 \
  --run-name formal_pureclip_l10_10x1000_seed0
```

Resume from the latest full checkpoint by adding:

```bash
--resume mmpartnet_out/film_runs/<run-name>/last.pt
```

The training script writes:

```text
best.pt            # compatibility name, same as Pearson-best
best_pearson.pt    # best validation profile Pearson
best_auprc.pt      # best validation binding AUPRC
last.pt            # latest full checkpoint, resumable
last.statedict.pt  # latest model weights only
metrics.json       # training/validation history
```

## How To Evaluate

Use validation for model selection:

```bash
cd /home/dgu/workspace/ML4RG_mmparnet_film

ML4RG_REFS=/home/dgu/workspace/parnet_refs \
ML4RG_PARNET_WEIGHTS=/home/dgu/storage_ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt \
PYTHONPATH=src \
/home/dgu/venvs/torch39/bin/python scripts/eval_film_multitask.py \
  --checkpoint mmpartnet_out/film_runs/<run-name>/best_pearson.pt \
  --split valid \
  --max-batches 2000 \
  --task multitask \
  --device cuda \
  --include-short \
  --out mmpartnet_out/film_runs/<run-name>/eval_valid_2000_best_pearson.json
```

Do not use the test split until the model design, checkpoint choice, and
hyperparameters are fixed.
