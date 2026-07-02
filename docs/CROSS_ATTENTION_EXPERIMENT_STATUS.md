# Cross-Attention Experiment Status

## Goal

The immediate goal was to check whether the multitask cross-attention head is
actually useful compared with single-task variants.

All three initial experiments used the same cross-attention backbone structure.
The main difference was the output head and the supervised loss:

1. `multitask`

   This is the full design described in the GitHub branch. It uses both a
   profile head and a binary head. In the multitask binary head, binding is
   predicted from a gated mixture of:

   - `target_prob` predicted by the profile head
   - `binary_position_prob` learned by the binary head

2. `profile-only`

   This keeps the same upstream cross-attention structure, but only keeps and
   trains the profile head.

3. `binary-only`

   This keeps the same upstream cross-attention structure, but only keeps and
   trains the binary head. In this mode, the binary head does not use
   `target_prob`; it uses its own learned `binary_position_prob` for pooling
   and binding prediction.

## Initial 5-Epoch Validation Ablation

The first diagnostic comparison ran three 5-epoch experiments.

```text
multitask_l10:
  best valid Pearson = 0.3641
  best valid AUPRC   = 0.1089

profile-only:
  best valid Pearson = 0.3700

binary-only_l10:
  best/final valid AUPRC = 0.0252
  valid positive rate    = 0.0258
```

The initial result was not clearly favorable to the multitask model:

- `profile-only` slightly outperformed `multitask` on profile Pearson.
- `binary-only` essentially failed under the original setting.
- `multitask` had better binary AUPRC than `binary-only`, but this should not
  be over-interpreted because `binary-only` itself appeared to have failed.

The validation positive rate is important for interpreting AUPRC. Here,
`valid positive rate = 0.0258` means that only about 2.58% of validation
samples are binding-positive. For a highly imbalanced binary task, a random
ranking has expected AUPRC close to the positive rate. Therefore,
`binary-only_l10` with `valid AUPRC = 0.0252` was approximately random on the
validation set.

## Binary-Only Diagnostics

Because the first `binary-only` result was essentially at the validation
positive-rate baseline, the next question was whether the binary head, labels,
or data wiring were fundamentally broken.

### Tiny Overfit Diagnostic

A tiny balanced training subset was used to test whether `binary-only` could
overfit.

```text
best train AUPRC       = 0.9701
best train binary loss = 0.0794
final train AUPRC      = 0.9540
```

This suggests that the binary head, labels, and data wiring are not
fundamentally broken. The model can overfit a small binary-only training set.

### Full-Data-Style Binary-Only Diagnostics With Lower Learning Rate

The original full-data-style `binary-only` run used `lr = 1e-3`. Two additional
2-epoch diagnostics lowered the learning rate to `3e-4` and compared two
protein latent lengths.

```text
latent256:
  epoch 1 train AUPRC = 0.7253, valid AUPRC = 0.0829
  epoch 2 train AUPRC = 0.7113, valid AUPRC = 0.0678
  best valid AUPRC    = 0.0829

latent128:
  epoch 1 train AUPRC = 0.6493, valid AUPRC = 0.0457
  epoch 2 train AUPRC = 0.6929, valid AUPRC = 0.0757
  best valid AUPRC    = 0.0757
```

These results suggest that the original `binary-only` failure was likely at
least partly an optimization issue. With `lr = 3e-4`, the full-data-style
`binary-only` model starts learning, whereas the earlier run with `lr = 1e-3`
was close to random on validation and did not show healthy training behavior.

The `latent256` setting is still the preferred next default because it matches
the original design and had slightly better best validation AUPRC in this
diagnostic.

## Current Interpretation

The current interpretation is:

- The cross-attention profile path is functional.
- The original `binary-only` failure does not appear to be caused by completely
  broken labels, data wiring, or binary-head implementation.
- The original `binary-only` failure is more likely related to full-data
  optimization or training setup, especially the learning rate.
- Lowering the learning rate to `3e-4` substantially improved the `multitask`
  run. The 15-epoch `multitask` diagnostic now beats both the previous
  `multitask` result and the previous `profile-only` Pearson result.

## Multitask Lower-LR Diagnostic

The next diagnostic reran the actual `multitask` model with the lower learning
rate that made `binary-only` start learning.

Run name:

```text
multitask_l10_lr3e4_latent256_5x1000_seed0
```

Shared output path:

```text
/home/dgu/workspace/cross_attention_runs/multitask_l10_lr3e4_latent256_5x1000_seed0
```

Resolved storage path:

```text
/mnt/storage1/workspace/dgu/cross_attention_runs/multitask_l10_lr3e4_latent256_5x1000_seed0
```

Initial 5-epoch configuration:

```text
task = multitask
lr = 3e-4
protein_latent_len = 256
lambda_profile = 1
lambda_binary = 10
batch_size = 32
epochs = 5
steps_per_epoch = 1000
tracks = all
max_train_windows = 0
max_valid_windows = 0
valid_sample_size = 32000
include_short = true
profile_mask_source = binding
num_blocks = 1
```

Results:

```text
best valid Pearson  = 0.4131, epoch 5
best valid AUPRC    = 0.1855, epoch 5
final valid Pearson = 0.4131
final valid AUPRC   = 0.1855
final train Pearson = 0.3942
final train AUPRC   = 0.8384
```

This run improved on the previous 5-epoch diagnostics:

```text
previous multitask best valid Pearson = 0.3641
new multitask best valid Pearson      = 0.4131

previous multitask best valid AUPRC   = 0.1089
new multitask best valid AUPRC        = 0.1855

previous profile-only best Pearson    = 0.3700
new multitask best valid Pearson      = 0.4131
```

This result suggests that the earlier weak `multitask` performance was at least
partly an optimization issue. With `lr = 3e-4`, the multitask model improves
both the profile and binary validation metrics.

## Completed 15-Epoch Resume

The improved multitask diagnostic was resumed from the 5-epoch checkpoint to 15
total epochs.

Resume checkpoint:

```text
/home/dgu/workspace/cross_attention_runs/multitask_l10_lr3e4_latent256_5x1000_seed0/last.pt
```

Resume log:

```text
/home/dgu/workspace/cross_attention_runs/multitask_l10_lr3e4_latent256_5x1000_seed0.resume_to15.log
```

Resume command shape:

```text
same configuration as the 5-epoch run
--epochs 10
--resume /home/dgu/workspace/cross_attention_runs/multitask_l10_lr3e4_latent256_5x1000_seed0/last.pt
```

15-epoch results:

```text
best valid Pearson  = 0.43395, epoch 15
best valid AUPRC    = 0.23989, epoch 15
final valid Pearson = 0.43395
final valid AUPRC   = 0.23989
final train Pearson = 0.43361
final train AUPRC   = 0.87573
```

Validation trajectory:

```text
epoch  1: valid Pearson = 0.3624, valid AUPRC = 0.0933
epoch  5: valid Pearson = 0.4131, valid AUPRC = 0.1855
epoch 10: valid Pearson = 0.4296, valid AUPRC = 0.2119
epoch 11: valid Pearson = 0.4336, valid AUPRC = 0.2311
epoch 15: valid Pearson = 0.4339, valid AUPRC = 0.2399
```

Comparison with earlier diagnostics:

```text
old multitask_l10 best valid Pearson = 0.3641
old multitask_l10 best valid AUPRC   = 0.1089

5-epoch lower-LR multitask Pearson   = 0.4131
5-epoch lower-LR multitask AUPRC     = 0.1855

15-epoch lower-LR multitask Pearson  = 0.43395
15-epoch lower-LR multitask AUPRC    = 0.23989

old profile-only best Pearson        = 0.3700
```

This run shows that `lr = 3e-4` plus 15 epochs substantially improves the
cross-attention multitask model. Epoch 15 is the best epoch for both validation
Pearson and validation AUPRC, so the lower-LR run had not clearly saturated by
the 5-epoch checkpoint.

The late-epoch `binding_gate_mean` is around 0.55-0.62, indicating that the
multitask binary head increasingly uses the profile head's `target_prob` branch
in its gated positional pooling. This is a useful signal for the hypothesis
that the profile head can help the binary head, but it still needs fair
single-task lower-LR comparisons.

## Completed Lower-LR Single-Task Comparisons

The previous `profile-only` and `binary-only` ablations were run under older
settings, so they were not fully comparable with the improved lower-LR
multitask run. The fairer single-task comparisons were rerun with the same
full-data-style setup:

```text
task = profile-only, lr = 3e-4, protein_latent_len = 256, 15 epochs
task = binary-only,  lr = 3e-4, protein_latent_len = 256, 15 epochs
```

### Profile-Only Lower-LR 15-Epoch Run

Run name:

```text
profile_only_lr3e4_latent256_15x500_seed0
```

Configuration:

```text
task = profile-only
lr = 3e-4
protein_latent_len = 256
lambda_profile = 1
lambda_binary = 0
batch_size = 32
epochs = 15
steps_per_epoch = 500
balanced_train = true
balanced_pos_fraction = 1.0
tracks = all
max_train_windows = 0
max_valid_windows = 0
valid_sample_size = 32000
include_short = true
profile_mask_source = binding
num_blocks = 1
```

Results:

```text
best valid Pearson  = 0.48450, epoch 14
final valid Pearson = 0.48391, epoch 15
final train Pearson = 0.48609
```

This profile-only run is currently the strongest cross-attention profile result.
It outperforms the lower-LR 15-epoch multitask run on profile Pearson:

```text
profile-only best valid Pearson = 0.48450
multitask best valid Pearson    = 0.43395
```

This suggests that the current multitask coupling does not improve profile
prediction. The binary objective may be competing with, or regularizing away
from, the profile-only optimum under the current settings.

### Binary-Only Lower-LR 15-Epoch Run

Run name:

```text
binary_only_l10_lr3e4_latent256_15x1000_seed0
```

Configuration:

```text
task = binary-only
lr = 3e-4
protein_latent_len = 256
lambda_profile = 0
lambda_binary = 10
binary_pooling = position
batch_size = 32
epochs = 15
steps_per_epoch = 1000
balanced_train = true
balanced_pos_fraction = 0.5
tracks = all
max_train_windows = 0
max_valid_windows = 0
valid_sample_size = 32000
include_short = true
profile_mask_source = binding
num_blocks = 1
```

Results:

```text
best valid AUPRC  = 0.08400, epoch 1
final valid AUPRC = 0.04025, epoch 15
final train AUPRC = 0.62596
valid positive rate = 0.02578
```

The binary-only model learns some signal on the balanced training stream, but
validation AUPRC remains low and degrades with longer training. This suggests
that the current binary-only position-pooling head is not a strong standalone
model under the full-data setting.

### Updated Interpretation After Fair Single-Task Runs

The fairer lower-LR results separate the two tasks more clearly:

```text
profile prediction:
  profile-only best valid Pearson = 0.48450
  multitask best valid Pearson    = 0.43395

binary prediction:
  multitask best valid AUPRC      = 0.23989
  binary-only best valid AUPRC    = 0.08400
```

The profile-only result indicates that multitask coupling is not currently
helping profile prediction. However, the multitask result is much stronger than
binary-only on binary AUPRC. This is consistent with the hypothesis that the
multitask binary head benefits from the profile head's predicted `target_prob`.

The key open question is whether that benefit comes from useful profile-guided
positional evidence, from optimization differences, or from dataset-specific
in-distribution signals.

## Completed Binary Loss Weight, Pooling, And Target-Alpha Diagnostics

Because the binary-only run with `lambda_binary = 10` was weak and appeared
unstable on validation, a short 5-epoch diagnostic lowered the binary loss
weight to `lambda_binary = 1`. A second binary-only diagnostic changed the
binary pooling from learned position-softmax pooling to masked mean pooling.

A third 5-epoch multitask diagnostic tested whether the multitask binary head
should pool only with the profile head's predicted `target_prob`, instead of
the original gated mixture of `target_prob` and `binary_position_prob`.

### Completed Position-Pooling Lambda-1 Diagnostic

Run name:

```text
binary_only_position_lam1_lr3e4_latent256_5x1000_seed0
```

Configuration:

```text
task = binary-only
lr = 3e-4
protein_latent_len = 256
lambda_profile = 0
lambda_binary = 1
binary_pooling = position
batch_size = 32
epochs = 5
steps_per_epoch = 1000
valid_sample_size = 32000
include_short = true
```

Results:

```text
best valid AUPRC  = 0.05276, epoch 4
final valid AUPRC = 0.05198, epoch 5
final train AUPRC = 0.63285
```

Lowering `lambda_binary` from 10 to 1 did not fix the position-pooling
binary-only model. The result is below the earlier 2-epoch lower-LR diagnostic
and below the 15-epoch `lambda_binary = 10` best epoch. This makes the loss
scale a less likely sole explanation for the binary-only weakness.

### Completed Mean-Pooling Lambda-1 Diagnostic

Run name:

```text
binary_only_mean_lam1_lr3e4_latent256_5x1000_seed0
```

Configuration:

```text
task = binary-only
lr = 3e-4
protein_latent_len = 256
lambda_profile = 0
lambda_binary = 1
binary_pooling = mean
epochs = 5
steps_per_epoch = 1000
batch_size = 32
balanced_train = true
balanced_pos_fraction = 0.5
tracks = all
max_train_windows = 0
max_valid_windows = 0
valid_sample_size = 32000
include_short = true
profile_mask_source = binding
num_blocks = 1
```

Results:

```text
best valid AUPRC  = 0.09150, epoch 3
final valid AUPRC = 0.08164, epoch 5
final train AUPRC = 0.70186
```

Mean pooling improved over position pooling under the same `lambda_binary = 1`
setting:

```text
position pooling, lambda_binary = 1:
  best valid AUPRC = 0.05276

mean pooling, lambda_binary = 1:
  best valid AUPRC = 0.09150
```

This suggests that the learned position-softmax pooling is a difficult
standalone binary-only pooling mechanism. However, mean pooling is still far
below the multitask binary result, so changing the pooling method does not make
`binary-only` competitive.

### Completed Target-Alpha Multitask Diagnostic

Run name:

```text
multitask_targetalpha_l10_lr3e4_latent256_5x1000_seed0
```

Configuration:

```text
task = multitask
binary_alpha_source = target
binary_pooling = position
lr = 3e-4
protein_latent_len = 256
lambda_profile = 1
lambda_binary = 10
epochs = 5
steps_per_epoch = 1000
batch_size = 32
balanced_train = true
balanced_pos_fraction = 0.5
tracks = all
max_train_windows = 0
max_valid_windows = 0
valid_sample_size = 32000
include_short = true
profile_mask_source = binding
num_blocks = 1
```

Results:

```text
best valid Pearson  = 0.38956, epoch 5
final valid Pearson = 0.38956
final train Pearson = 0.38345

best valid AUPRC  = 0.15848, epoch 4
final valid AUPRC = 0.15224
final train AUPRC = 0.82081
```

Validation trajectory:

```text
epoch 1: valid Pearson = 0.3624, valid AUPRC = 0.1241
epoch 2: valid Pearson = 0.3746, valid AUPRC = 0.1292
epoch 3: valid Pearson = 0.3596, valid AUPRC = 0.1467
epoch 4: valid Pearson = 0.3792, valid AUPRC = 0.1585
epoch 5: valid Pearson = 0.3896, valid AUPRC = 0.1522
```

The target-alpha diagnostic did not outperform the original gated multitask
head at 5 epochs:

```text
gated multitask, 5 epochs:
  valid Pearson = 0.4131
  valid AUPRC   = 0.1855

target-alpha multitask, 5 epochs:
  valid Pearson = 0.3896
  best valid AUPRC = 0.1585
```

This result argues against the simplest version of the hypothesis that binary
prediction should use only `target_prob` from the profile head. The original
gated mixture is still better in this short comparison, suggesting that the
separate `binary_position_prob` branch and/or learned gate do contribute useful
information beyond the profile distribution.

## Current Summary After Completed Diagnostics

The updated comparison is:

```text
profile prediction:
  profile-only, 15 epochs:
    best valid Pearson = 0.48450

  gated multitask, 15 epochs:
    best valid Pearson = 0.43395

  target-alpha multitask, 5 epochs:
    best valid Pearson = 0.38956

binary prediction:
  gated multitask, 15 epochs:
    best valid AUPRC = 0.23989

  gated multitask, 5 epochs:
    best valid AUPRC = 0.18550

  target-alpha multitask, 5 epochs:
    best valid AUPRC = 0.15848

  binary-only mean, lambda_binary = 1, 5 epochs:
    best valid AUPRC = 0.09150

  binary-only position, lambda_binary = 10, 15 epochs:
    best valid AUPRC = 0.08400

  binary-only position, lambda_binary = 1, 5 epochs:
    best valid AUPRC = 0.05276
```

The strongest current conclusions are:

- The cross-attention profile path is useful. The strongest profile result so
  far is `profile-only`, not `multitask`.
- The current binary-only heads generalize weakly, even though tiny overfit
  showed that the binary wiring is not fundamentally broken.
- Lowering `lambda_binary` from 10 to 1 does not fix binary-only.
- Mean pooling is better than learned position-softmax pooling for binary-only,
  but still far below multitask.
- Multitask substantially improves binary AUPRC relative to binary-only, but it
  hurts profile Pearson relative to profile-only.
- Target-alpha pooling does not beat the original gated multitask design in the
  5-epoch comparison.

At this point, the profile task appears to be the clearer strength of the
cross-attention model. The binary task remains useful as an auxiliary or
evaluation question, but the current binary-only formulation is not a strong
standalone model.
