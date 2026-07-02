# Cross-Attention Workflow

This branch adds a protein-residue cross-attention baseline on top of the FiLM
multitask setup. The current head is a cell-FiLM bidirectional cross-attention
model: each block conditions RNA tokens on cell line, updates RNA and protein
tokens with synchronous multi-head cross-attention, then a final RNA-to-protein
attention pass produces the RNA-centric representation used by the output heads.

## Model Variants

Two cross-attention heads live side by side and share the same data loader,
sampler, loss, and metrics, so they can be benchmarked directly. Select with
`--model` in the training/eval scripts (default `original`):

- `original` — `ProteinCellCrossAttentionProfileHead`
  (`src/mmpartnet/models/cross_attention.py`): the cell-FiLM bidirectional head
  with the `alpha_bind` gated binary output.
- `tfbind` — `TFBindCrossAttentionProfileHead`
  (`src/mmpartnet/models/cross_attention_tfbind.py`): a TFBindFormer-style
  rebuild. Key differences to the original:
  1. **ProteinCompressor** — compresses variable-length ProtT5 residues to a
     fixed `target_prot_len` via learned queries (no per-batch protein padding
     downstream).
  2. **Asymmetric bidirectionality** — only the first `num_bidir_blocks` blocks
     update both streams; later blocks let RNA query protein only.
  3. **PositionWeightedPool** — a single learned attention pooling over RNA
     positions feeds the binary head, replacing the `alpha_bind` / `binding_gate`
     construction.

The `tfbind` head is frozen-PARNET + ~1.2M trainable parameters.

## Branch

```bash
git checkout dgu/cross-attention
```

The branch should keep the FiLM implementation intact and add separate
cross-attention files:

- `src/mmpartnet/models/cross_attention.py`
- `scripts/train_cross_attention_profile.py`
- `scripts/eval_cross_attention_profile.py`
- `scripts/sanity_cross_attention_model.py`

## Data

Use the residue-level ProtT5 H5 for cross-attention:

```text
/mnt/storage1/ml4rg26-mmparnet/manually_gathered/ProtT5_zenodo_datasets/embeddings_file.h5
```

Use the pooled H5 only for FiLM:

```text
/mnt/storage1/ml4rg26-mmparnet/manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5
```

The cross-attention collator pads protein residue embeddings in each batch and
returns:

- `protein_residue_embedding`: `[B, Lp, 1024]`
- `protein_mask`: `[B, Lp]`
- `protein_embedding`: pooled over the residue tokens for compatibility

For memory-sensitive runs, pass `--max-protein-len N` to truncate residue tokens.
Among the 223 matched PARNET tracks, the longest mapped protein is currently
3323 residues and the median is 465 residues.

## Model Sketch

```text
RNA one-hot -> frozen PARNET body -> RNA tokens [B, 600, H]
ProtT5 residues -> projection -> protein tokens [B, Lp, H]
cell id -> cell embedding

for each block:
  RNA = Cell-FiLM(RNA, cell)
  protein = Protein attends RNA
  RNA = RNA attends protein

final:
  RNA = RNA attends protein

profile head:
  RNA -> target/control/total profiles + mix coefficient

binary head:
  binary_prob = softmax(Linear(RNA))
  gate = sigmoid(MLP(masked_mean(RNA)))
  alpha_bind = gate * target_prob + (1 - gate) * binary_prob
  weighted RNA summary -> binding logit
```

`forward()` returns `binding_gate`, `binary_position_prob`, and `alpha_bind` for
interpretability.

Training and evaluation metrics include lightweight distribution summaries:

- gate mean/std, plus positive/negative gate means
- entropy, max probability, and top-10 probability mass for `target`,
  `binary_position_prob`, and `alpha_bind`

Use the export script below for full per-sample length-600 distributions.

## Smoke Test

```bash
.venv/bin/python scripts/sanity_cross_attention_model.py
```

## Small Training Run

```bash
.venv/bin/python scripts/train_cross_attention_profile.py \
  --tracks 9,138,195 \
  --max-train-windows 256 \
  --max-valid-windows 128 \
  --batch-size 4 \
  --epochs 1 \
  --max-protein-len 1024 \
  --num-blocks 1 \
  --run-name smoke_cross_attention
```

## Formal Validation Run

Do not use the test split for model selection. Match the FiLM convention and
select by validation Pearson or validation AUPRC.

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
  --max-protein-len 1024 \
  --num-blocks 1 \
  --run-name formal_pureclip_cross_attention_l10_15x1000_seed0
```

## Evaluate A Checkpoint

```bash
.venv/bin/python scripts/eval_cross_attention_profile.py \
  --checkpoint mmpartnet_out/cross_attention_runs/formal_pureclip_cross_attention_l10_15x1000_seed0/best_pearson.pt \
  --split valid \
  --max-windows 2000 \
  --batch-size 8
```

## Export Interpretation Samples

```bash
.venv/bin/python scripts/export_cross_attention_interpretation.py \
  --checkpoint mmpartnet_out/cross_attention_runs/formal_pureclip_cross_attention_l10_15x1000_seed0/best_pearson.pt \
  --split valid \
  --max-samples 256 \
  --batch-size 8 \
  --out mmpartnet_out/cross_attention_runs/formal_pureclip_cross_attention_l10_15x1000_seed0/interpretation_valid.pt
```

If a motif TSV is available, pass it to compute per-sample motif overlap metrics:

```bash
.venv/bin/python scripts/export_cross_attention_interpretation.py \
  --checkpoint mmpartnet_out/cross_attention_runs/formal_pureclip_cross_attention_l10_15x1000_seed0/best_pearson.pt \
  --split valid \
  --max-samples 256 \
  --motif-tsv path/to/rbp_motifs.tsv \
  --out mmpartnet_out/cross_attention_runs/formal_pureclip_cross_attention_l10_15x1000_seed0/interpretation_valid.pt \
  --motif-out mmpartnet_out/cross_attention_runs/formal_pureclip_cross_attention_l10_15x1000_seed0/motif_eval_valid.tsv
```

The motif TSV must contain `rbp` and `motif` columns. Motifs may use RNA or DNA
letters; common IUPAC ambiguity codes are supported.

## Benchmark: original vs tfbind

`scripts/compare_cross_attention_models.py` trains both heads over several seeds
(as subprocesses of the trainer, so data/sampler/metrics are byte-identical),
reads each run's `metrics.json`, and prints a mean ± std table of the
best-over-epochs validation values.

```bash
.venv/bin/python scripts/compare_cross_attention_models.py \
  --seeds 0 1 2 \
  --max-train-windows 2000 --max-valid-windows 1000 \
  --tracks 9,138,195 --epochs 5 --batch-size 8 \
  --steps-per-epoch 400 --balanced-train
```

### Result (seeds 0/1/2, tracks 9,138,195, 2000 train windows, 5 epochs)

Validation, best-over-epochs, mean ± std:

| Metric                  | original         | tfbind           |
| ----------------------- | ---------------- | ---------------- |
| profile Pearson (up)    | 0.360 ± 0.007    | **0.398 ± 0.015** |
| binding AUPRC (up)      | 0.140 ± 0.012    | 0.154 ± 0.010    |
| valid loss (down)       | 13.89 ± 1.01     | 13.27 ± 0.77     |

Reading:

- **Profile Pearson**: `tfbind` wins cleanly. The ± bands do not overlap, and
  every `tfbind` seed (0.389 / 0.385 / 0.420) beats every `original` seed
  (0.351 / 0.369 / 0.361) — the worst `tfbind` run still beats the best
  `original` run. This is the Milestone-2 target metric.
- **Binding AUPRC**: roughly tied (bands overlap). The binary-head overfitting
  seen in the earlier 256-window smoke test disappeared with more data.
- **Caveats**: only 3 tracks, one data slice, n=3 seeds, and this is
  **same-RBP** validation — it does not yet test generalization to unseen RBPs
  (the leave-one-RBP-out setting, pending leave-out PARNET weights).

The scaled-up run (all matched tracks, more windows) is the next step to confirm
the profile-Pearson advantage holds across all 223 RBPs.

## Notes

- `multimodal`: RNA + residue-level protein + cell.
- `rna-only`: zeros protein tokens and cell ids.
- `no-cell`: uses protein tokens but zeros cell ids.
- `protein-shuffle`: shuffles protein tokens across the batch.

The main architectural path is:

```text
RNA one-hot -> frozen PARNET body -> [B, C, 600]
ProtT5 residues -> [B, Lp, 1024]
RNA positions query protein residues with cell-conditioned cross-attention
fused RNA tokens -> profile head + binary binding head
```
