# Cross-Attention Workflow

This branch adds a protein-residue cross-attention baseline on top of the FiLM
multitask setup. The current head is a cell-FiLM bidirectional cross-attention
model: each block conditions RNA tokens on cell line, updates RNA and protein
tokens with synchronous multi-head cross-attention, then a final RNA-to-protein
attention pass produces the RNA-centric representation used by the output heads.

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
