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

## Next Fair Comparisons

The previous `profile-only` and `binary-only` ablations were run under older
settings, so they are no longer fully comparable with the improved lower-LR
multitask run. The next fair single-task comparisons should use the same
full-data-style setup:

```text
task = profile-only, lr = 3e-4, protein_latent_len = 256, 15 epochs
task = binary-only,  lr = 3e-4, protein_latent_len = 256, 15 epochs
```

These runs are needed to distinguish:

```text
better optimization of the shared cross-attention backbone
vs.
actual benefit from coupling the profile and binary heads
```
