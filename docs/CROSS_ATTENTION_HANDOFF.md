# Cross-Attention Handoff For Another Codex

This document is a practical handoff for continuing the cross-attention branch
on another VM. It explains what the branch is doing, how the training loop
works, what commands to run, and what details should not be accidentally
changed.

## Current Branch

```bash
git checkout dgu/cross-attention
git pull origin dgu/cross-attention
```

Main files:

- `src/mmpartnet/models/cross_attention.py`
- `src/mmpartnet/data/multimodal.py`
- `scripts/train_cross_attention_profile.py`
- `scripts/eval_cross_attention_profile.py`
- `scripts/export_cross_attention_interpretation.py`
- `scripts/sanity_cross_attention_model.py`
- `docs/CROSS_ATTENTION_WORKFLOW.md`

Do not rewrite the FiLM baseline destructively. This branch adds a separate
cross-attention model and scripts.

## Task

The model predicts two related outputs for one flattened sample:

```text
RNA window + RBP protein + cell line -> binding / not-binding
RNA window + RBP protein + cell line -> binding signal profile
```

The original PARNET predicts all RBP-cell tracks from RNA sequence alone. This
branch flattens the task so that one sample is:

```text
one RNA window + one RBP-cell track
```

For example:

```text
QKI_HepG2 and QKI_K562 are different tracks.
They share the same RBP name but have different cell-line context and labels.
```

## Data Paths

On the current VM, the defaults in the scripts point through:

```text
/home/dgu/storage_ml4rg26-mmparnet
```

Important datasets:

```text
PARNET HFDS:
/mnt/storage1/ml4rg26-shared/parnet-eclip/data-formatted-for-training/600nt_windows.no-one-hot.stripped/encode.filtered.hfds

PureCLIP binary binding labels:
/mnt/storage1/ml4rg26-mmparnet/manually_gathered/600nt_windows.no-one-hot.stripped.binding/600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt

Residue-level ProtT5 H5:
/mnt/storage1/ml4rg26-mmparnet/manually_gathered/ProtT5_zenodo_datasets/embeddings_file.h5

Pooled ProtT5 H5, only for FiLM-style baselines:
/mnt/storage1/ml4rg26-mmparnet/manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5

Track-to-ProtT5 map:
mmpartnet_out/prott5_track_map.tsv
```

The training/eval scripts accept `--hfds`, `--binding-dataset`,
`--protein-h5`, and `--track-map` if a different VM uses different mount paths.

## Model Summary

The current model is `ProteinCellCrossAttentionProfileHead`.

Input shapes:

```text
RNA one-hot:                  [B, 4, 600]
PARNET body RNA features:     [B, 512, 600]
RNA tokens after transpose:   [B, 600, 512]

ProtT5 residue embeddings:    [B, Lp, 1024]
Protein mask:                 [B, Lp]

Cell index:                   [B]
Cell embedding:               [B, 32]
```

Protein path:

```text
ProtT5 residues: [B, Lp, 1024]
-> Linear(1024, 768)
-> GELU
-> Dropout(0.1)
-> Linear(768, 512)
-> LayerNorm
-> latent-query compression
compressed protein tokens: [B, 256, 512]
```

Latent-query compression:

```text
learned latent queries: [256, 512]
Q = latent queries
K = protein residue tokens
V = protein residue tokens
MultiHeadAttention(Q, K, V) -> [B, 256, 512]
```

Padding residues are masked, so latent queries cannot attend to padded
positions.

Cell path:

```text
cell_index: [B]        # current data: 0 = HepG2, 1 = K562
-> Embedding(2, 32)
cell embedding: [B, 32]
```

The model also supports `cell_index = -1`, which means zero cell embedding.
This is used for `--mode no-cell` and `--mode rna-only`.

Fusion block:

```text
RNA = Cell-FiLM(RNA, cell)
protein_next = Protein attends RNA
RNA_next     = RNA attends protein
```

Each cross-attention update uses:

```text
multi-head cross-attention
-> residual
-> LayerNorm
-> FFN
-> residual
-> LayerNorm
```

After `num_blocks`, there is one final asymmetric update:

```text
RNA attends protein
```

This gives the final RNA-centric representation:

```text
Z: [B, 600, 512]
```

## Output Heads

Profile head:

```text
target_logits  = Linear(Z) -> [B, 600]
control_logits = Linear(Z) -> [B, 600]

target_prob  = softmax(target_logits)
control_prob = softmax(control_logits)
```

Mixture coefficient:

```text
pooled = masked_mean(Z)                 # [B, 512]
mix_coeff = sigmoid(MLP(pooled))        # [B]
total_profile = mix_coeff * target_prob + (1 - mix_coeff) * control_prob
```

Binary head:

```text
binary_prob = softmax(Linear(Z))        # [B, 600]
gate = sigmoid(MLP(masked_mean(Z)))     # [B]

alpha_bind = gate * target_prob + (1 - gate) * binary_prob
v_bind = sum_i alpha_bind_i * z_i       # [B, 512]
binding_logit = MLP(v_bind)             # [B]
```

Important: `target_prob` is the model-predicted target profile distribution. It
is not the observed eCLIP count distribution.

`target_prob` is not detached before entering the binary head. Therefore binary
loss can backpropagate through `target_prob` into the target-profile branch.

## Loss

Training computes:

```text
loss = profile_loss + lambda_binary * binary_loss
```

with optional:

```text
loss += mix_penalty * mean(mix_coeff)
```

Profile loss:

```text
profile_loss = eCLIP profile NLL + control profile NLL
```

Current default profile supervision:

```text
--profile-mask-source binding
profile_keep = binding_label > 0.5
```

So by default:

```text
PureCLIP-positive samples contribute profile_loss.
All labeled samples contribute binary_loss.
```

Other available profile masks:

```text
--profile-mask-source count
--profile-mask-source binding-and-count
```

Binary loss:

```text
BCEWithLogitsLoss(binding_logit, binding_label)
```

## Training Logic

One training epoch does the following:

1. Build flattened samples from PARNET windows and selected RBP-cell tracks.
2. Collate RNA one-hot, eCLIP/control counts, PureCLIP binding labels, cell
   index, padded residue-level protein embeddings, and protein masks.
3. Run frozen PARNET body:

   ```text
   rna_features = parnet.body_feats(onehot).detach()
   ```

   PARNET is frozen; only the cross-attention head is trained.

4. Apply optional mode:

   ```text
   multimodal       RNA + protein + cell
   no-cell          RNA + protein + zero cell embedding
   rna-only         RNA + zero protein embeddings + zero cell embedding
   protein-shuffle  RNA + shuffled protein within batch + original cell
   ```

5. Build `profile_mask` from `--profile-mask-source`.
6. Forward the cross-attention head.
7. Compute profile loss and binary loss.
8. Backpropagate into the head only.
9. During validation, compute:

   ```text
   profile Pearson
   binding accuracy
   binding AUPRC
   binding_gate mean/std
   positive/negative gate means
   entropy, max probability, top10 mass for target/binary/alpha distributions
   ```

10. Save:

   ```text
   best_pearson.pt
   best_auprc.pt
   best.pt
   last.pt
   last.statedict.pt
   metrics.json
   ```

Do not use the test split for model selection. Use validation for selecting
checkpoint and hyperparameters.

## Quick Sanity Checks

Run these after pulling the branch:

```bash
PYTHONPYCACHEPREFIX=/tmp/mmpartnet_pycache \
  .venv/bin/python -m py_compile \
  src/mmpartnet/models/cross_attention.py \
  scripts/train_cross_attention_profile.py \
  scripts/eval_cross_attention_profile.py \
  scripts/export_cross_attention_interpretation.py \
  scripts/sanity_cross_attention_model.py
```

```bash
.venv/bin/python scripts/sanity_cross_attention_model.py --protein-len 120
.venv/bin/python scripts/sanity_cross_attention_model.py --protein-len 350
```

The sanity script checks:

```text
target/control/total profile shapes and sums
binary_position_prob and alpha_bind sums
binding logits
loss finite
cell_index = -1 zero-cell path
```

## Smoke Training Command

This is a very small CPU/GPU smoke run. Use `/tmp` for output if the repo output
directory is not writable on the VM.

```bash
.venv/bin/python scripts/train_cross_attention_profile.py \
  --tracks 9 \
  --max-train-windows 4 \
  --max-valid-windows 4 \
  --batch-size 2 \
  --epochs 1 \
  --max-train-batches 1 \
  --max-valid-batches 1 \
  --progress-every 1 \
  --out-dir /tmp/mmpartnet_cross_attention_runs \
  --run-name latent_query_smoke
```

Then test checkpoint loading:

```bash
.venv/bin/python scripts/eval_cross_attention_profile.py \
  --checkpoint /tmp/mmpartnet_cross_attention_runs/latent_query_smoke/best_pearson.pt \
  --split valid \
  --max-windows 4 \
  --batch-size 2 \
  --max-batches 1
```

## Suggested Formal Validation Run

This is the first formal run to compare against FiLM baselines. Keep test split
untouched.

```bash
.venv/bin/python scripts/train_cross_attention_profile.py \
  --tracks all \
  --max-train-windows 0 \
  --max-valid-windows 0 \
  --batch-size 8 \
  --epochs 15 \
  --balanced-train \
  --steps-per-epoch 1000 \
  --balanced-pos-fraction 0.5 \
  --lambda-binary 10 \
  --profile-mask-source binding \
  --num-blocks 1 \
  --run-name formal_pureclip_cross_attention_l10_15x1000_seed0
```

If GPU memory is tight, try:

```text
--batch-size 4
--protein-latent-len 128
```

The current default uses:

```text
hidden_dim = 512
num_heads = 8
protein_projection_hidden_dim = 768
protein_compression = latent
protein_latent_len = 256
cell_dim = 32
dropout = 0.1
lambda_binary = 20 by script default, but use 10 for first formal comparison
```

## Evaluation

```bash
.venv/bin/python scripts/eval_cross_attention_profile.py \
  --checkpoint path/to/best_pearson.pt \
  --split valid \
  --max-windows 2000 \
  --batch-size 8 \
  --out path/to/eval_valid_2000.json
```

Useful ablation modes:

```bash
--mode multimodal
--mode no-cell
--mode rna-only
--mode protein-shuffle
```

Interpret modes carefully:

- `no-cell` uses a true zero cell embedding.
- `rna-only` zeros residue embeddings and uses zero cell embedding, but the
  latent compression module still produces learned null protein summary tokens
  from the zero input. This removes protein identity, but it does not bypass the
  protein-attention pathway entirely.
- `protein-shuffle` shuffles protein embeddings within the batch.

## Interpretation Export

```bash
.venv/bin/python scripts/export_cross_attention_interpretation.py \
  --checkpoint path/to/best_pearson.pt \
  --split valid \
  --max-samples 256 \
  --batch-size 8 \
  --out path/to/interpretation_valid.pt
```

With motif metrics:

```bash
.venv/bin/python scripts/export_cross_attention_interpretation.py \
  --checkpoint path/to/best_pearson.pt \
  --split valid \
  --max-samples 256 \
  --batch-size 8 \
  --motif-tsv path/to/rbp_motifs.tsv \
  --out path/to/interpretation_valid.pt \
  --motif-out path/to/motif_eval_valid.tsv
```

Motif TSV format:

```text
rbp<TAB>motif
QKI<TAB>ACUAAY
```

The export includes:

```text
target_prob
binary_position_prob
alpha_bind
binding_gate
binding_prob
pred_control
pred_total
mix_coeff
true eCLIP/control counts
sequence and metadata
optional motif_mask
```

For motif interpretability, compare:

```text
target_prob_on_motif
binary_prob_on_motif
alpha_bind_on_motif
topk_target_overlap_motif
topk_binary_overlap_motif
topk_alpha_overlap_motif
```

## Code Review Notes From This VM

Validated in the current workspace:

```text
py_compile passed
sanity_cross_attention_model.py --protein-len 120 passed
sanity_cross_attention_model.py --protein-len 350 passed
tiny no-cell train smoke passed
tiny no-cell checkpoint eval passed
```

One issue was found and fixed:

```text
Old no-cell mode used cell_index = 0, which means HepG2, not zero-cell.
The model now treats cell_index = -1 as zero cell embedding.
train_cross_attention_profile.apply_mode now uses -1 for no-cell and rna-only.
```

No obvious tensor-shape or loss-contract issue was found in the default
multimodal path after this fix.

## Things To Avoid

- Do not use the test split for model selection.
- Do not replace `target_prob` with observed eCLIP counts in the binary head.
  The binary head intentionally uses the model-predicted target distribution.
- Do not detach `target_prob` unless intentionally testing an ablation where
  binary loss is prevented from updating the target-profile branch.
- Do not destructively edit FiLM baseline files.
- Be careful interpreting `profile_loss = 0` in tiny smoke runs; it often means
  the sampled batch had no profile-kept examples.

## Natural Next Experiments

- `lambda_binary`: compare 5, 10, 20.
- `num_blocks`: compare 1 vs 2 after memory check.
- `protein_latent_len`: compare 128 vs 256.
- `profile_mask_source`: compare `binding` vs `binding-and-count`.
- gated binary pooling ablation:

  ```text
  alpha_bind = target_prob only
  alpha_bind = binary_prob only
  alpha_bind = gated mixture
  ```

- cell conditioning ablation:

  ```text
  RNA FiLM only
  protein FiLM only
  RNA + protein FiLM
  cell token in attention
  ```
