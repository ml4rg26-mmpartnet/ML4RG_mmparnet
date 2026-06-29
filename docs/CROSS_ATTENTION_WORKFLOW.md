# Cross-Attention Model Design

This branch explores a residue-level protein/RNA/cell fusion model for
multimodal PARNET. The goal is to predict both:

```text
RNA window + RBP protein + cell line -> binding / not-binding
RNA window + RBP protein + cell line -> binding signal profile
```

The original PARNET predicts all RBP-cell tracks from RNA sequence alone. In
this branch, each sample is one RNA window paired with one RBP-cell track. The
model therefore needs to represent three inputs at the same time: RNA sequence,
protein identity/sequence, and cell line context.

## Branch And Main Files

```bash
git checkout dgu/cross-attention
```

- `src/mmpartnet/models/cross_attention.py`
- `scripts/train_cross_attention_profile.py`
- `scripts/eval_cross_attention_profile.py`
- `scripts/export_cross_attention_interpretation.py`
- `scripts/sanity_cross_attention_model.py`

## Data Inputs

The cross-attention model uses residue-level protein embeddings rather than one
pooled protein vector. The collator returns:

- `onehot`: RNA one-hot sequence, `[B, 4, 600]`
- `mask`: valid RNA positions, `[B, 600]`
- `protein_residue_embedding`: padded residue-level ProtT5 features, `[B, Lp, 1024]`
- `protein_mask`: valid protein residues, `[B, Lp]`
- `cell_index`: learned cell-line id, `[B]`
- `binding`: PureCLIP binary binding label, `[B]`
- `eclip` and `control`: profile count targets, `[B, 600]`

## Protein Feature Projection

The protein input starts as residue-level ProtT5 embeddings:

```text
protein residue embedding: [B, Lp, 1024]
```

Cross-attention is easiest to implement and reason about when RNA tokens and
protein tokens share the same feature dimension. The frozen PARNET body gives
RNA features with 512 channels, so the protein representation is projected to
512 dimensions:

```text
protein residue embedding: [B, Lp, 1024]
-> Linear(1024, 768)
-> GELU
-> Dropout(0.1)
-> Linear(768, 512)
-> LayerNorm
protein tokens: [B, Lp, 512]
```

This is a two-layer MLP rather than a single linear projection because a single
linear layer can only remap the feature space linearly. Adding `GELU` gives the
projection a non-linear step, which should help the model adapt ProtT5 features
to the RNA/protein fusion space. The final `LayerNorm` stabilizes the scale of
the protein tokens before they enter multi-head attention.

The RNA representation comes from the frozen PARNET body:

```text
RNA one-hot: [B, 4, 600]
-> frozen PARNET body
RNA features: [B, 512, 600]
-> transpose
RNA tokens: [B, 600, 512]
-> LayerNorm
RNA tokens: [B, 600, 512]
```

Because RNA is already 512-dimensional, the current implementation does not add
an extra RNA projection. It only applies `LayerNorm` before fusion so that the
RNA and protein token scales are better matched.

## Protein Length Compression

Protein lengths are highly variable. Direct RNA/protein cross-attention has a
cost that scales with RNA length times protein length, so very long proteins can
be expensive. In our task, the final outputs are binding probability and RNA
position profiles. We need the protein to condition the RNA representation, but
we do not need to keep one output token per original residue all the way through
the model. For this reason, the current model compresses the protein sequence
length after the feature projection.

Track-weighted protein length statistics for the 223 matched PARNET tracks:

| statistic | protein length |
|---|---:|
| min | 58 |
| 10th percentile | 130 |
| 25th percentile | 271 |
| median | 465 |
| 75th percentile | 688 |
| 90th percentile | 1004 |
| 95th percentile | 1268 |
| max | 3323 |
| mean | 539.3 |

Threshold counts:

| threshold | tracks above threshold |
|---|---:|
| >128 residues | 202 / 223 |
| >256 residues | 170 / 223 |
| >512 residues | 93 / 223 |
| >1024 residues | 18 / 223 |
| >2048 residues | 3 / 223 |

The current design uses learned latent queries, following the same general idea
as TFBindFormer's protein reduction step.

The idea is:

```text
protein tokens: [B, Lp, 512]
learned latent queries: [256, 512]
```

The 256 latent queries are trainable vectors. They are not residues. Instead,
they act like 256 learned "summary slots". During attention, each latent query
looks over all real protein residue tokens and learns what kind of protein
information it should collect.

In attention notation:

```text
Q = learned latent queries
K = protein residue tokens
V = protein residue tokens

compressed protein = MultiHeadAttention(Q, K, V)
```

The output length is the query length, so the compressed protein always has 256
tokens:

```text
protein tokens: [B, Lp, 512]
-> latent-query compression
compressed protein tokens: [B, 256, 512]
```

Padding residues are masked, so latent queries cannot read padded positions.

Overall protein path:

```text
ProtT5 residues: [B, Lp, 1024]
-> protein MLP: [B, Lp, 512]
-> latent-query compression: [B, 256, 512]
```

## Cell Representation

The same RBP can behave differently in different cell lines, so cell context is
part of the conditioning signal. The model uses a learned cell embedding:

```text
cell_index: [B]
-> Embedding
cell vector: [B, cell_dim]
```

The current cell vocabulary is built from the RBP-cell track names, such as
`QKI_HepG2` and `QKI_K562`.

## Fusion Block

Each fusion block combines the three modalities once:

```text
RNA tokens: [B, 600, 512]
protein tokens: [B, 256, 512]
cell vector: [B, cell_dim]
```

First, the cell vector conditions the RNA tokens with FiLM:

```text
gamma, beta = Linear(cell)
RNA_cell = (1 + tanh(gamma)) * RNA + beta
```

This is applied to RNA rather than protein in the first version because the
protein sequence is the same across cell lines, while cell line effects are
more likely to change the RNA/eCLIP context, transcript environment, and
observed binding signal. This is a design choice, not a claim that cell never
affects protein behavior. Good future ablations include FiLM on protein as
well, adding cell tokens to attention, or conditioning both modalities.

After cell-FiLM, RNA and protein update each other with synchronous multi-head
cross-attention:

```text
protein_next = Protein attends cell-conditioned RNA
RNA_next     = cell-conditioned RNA attends protein
```

Both directions use standard Transformer-style updates:

```text
cross-attention
-> residual add
-> LayerNorm
-> feed-forward network
-> residual add
-> LayerNorm
```

One block therefore lets RNA, protein, and cell information interact once. The
number of blocks is configurable:

```text
--num-blocks 1
--num-blocks 2
--num-blocks 3
```

The recommended first setting is `num_blocks = 1`, then increase the number of
blocks only after the model is training stably.

## Final RNA-Centric Representation

After all bidirectional fusion blocks, the model applies one final asymmetric
cross-attention:

```text
RNA attends protein
```

This produces the final RNA-centric representation:

```text
Z: [B, 600, 512]
```

The representation is RNA-centric because the profile task is defined over RNA
positions. Each RNA position should carry its own local RNA information plus
protein-aware and cell-aware context.

## Output Heads

The model has two output heads: a profile head and a binary binding head.

### Profile Head

The profile head predicts target and control position distributions:

```text
Z: [B, 600, 512]

target_logits  = Linear(Z)  -> [B, 600]
control_logits = Linear(Z)  -> [B, 600]

target_prob  = softmax(target_logits)  -> [B, 600]
control_prob = softmax(control_logits) -> [B, 600]
```

The model also predicts a mixture coefficient:

```text
mix_coeff = sigmoid(MLP(masked_mean(Z))) -> [B]
```

The final predicted eCLIP profile is a mixture of target and control profiles:

```text
total_profile = mix_coeff * target_prob + (1 - mix_coeff) * control_prob
```

The profile loss is:

```text
profile_loss = eCLIP profile NLL + control profile NLL
```

By default, profile loss is only applied to PureCLIP-positive binding samples:

```text
profile_mask_source = "binding"
profile_keep = binding > 0.5
```

Here `binding > 0.5` is just a robust way to select labels equal to 1 when the
label tensor is stored as float `0.0`/`1.0`.

### Binary Binding Head

The binary head predicts whether the RNA window binds the RBP-cell track. It
uses position-weighted pooling over the final RNA-centric representation.

The binary head first learns its own position distribution:

```text
binary_score = Linear(Z)              -> [B, 600]
binary_prob  = softmax(binary_score)  -> [B, 600]
```

The profile head already predicts `target_prob`, which can be interpreted as
where the model thinks the protein-specific signal lies on the RNA. This is
useful for binary classification, but using it alone is risky because a softmax
distribution always sums to 1, even for negative samples. Therefore the binary
head uses a gate to mix profile-based pooling and binary-specific pooling:

```text
gate = sigmoid(MLP(masked_mean(Z))) -> [B]

alpha_bind = gate * target_prob + (1 - gate) * binary_prob
```

Then the model pools RNA positions using `alpha_bind`:

```text
v_bind = sum_i alpha_bind_i * z_i -> [B, 512]
binding_logit = MLP(v_bind)       -> [B]
binding_prob = sigmoid(binding_logit)
```

This design keeps the biological intuition that profile peaks should help
binding classification, while still allowing the binary head to learn its own
position weighting when the predicted profile is not reliable.

### Total Loss

The total training loss is:

```text
total_loss = profile_loss + lambda_binary * binary_loss
```

with optional `mix_penalty`:

```text
total_loss = profile_loss + lambda_binary * binary_loss
           + mix_penalty * mean(mix_coeff)
```

The default binary loss is:

```text
binary_loss = BCEWithLogitsLoss(binding_logit, binding_label)
```

Important detail:

```text
All samples go through the profile head and produce target_prob.
Only profile_keep samples contribute profile_loss.
All labeled samples contribute binary_loss.
```

Because `target_prob` is used inside the binary head, binary loss can still send
gradient through the target-profile branch even for negative samples.

## Interpretability Outputs

The forward pass returns:

- `target`: predicted target profile distribution, `[B, 600]`
- `binary_position_prob`: binary head's own position distribution, `[B, 600]`
- `alpha_bind`: final mixed pooling distribution, `[B, 600]`
- `binding_gate`: how much the binary head relies on `target_prob` vs `binary_prob`, `[B]`
- `binding_prob`: final binding probability, `[B]`

These are the main tensors to compare against known motifs. The export script
saves full per-sample distributions and optional motif overlap metrics.

## Main Architecture Summary

```text
RNA one-hot -> frozen PARNET body -> RNA tokens [B, 600, 512]

ProtT5 residues [B, Lp, 1024]
-> protein MLP [B, Lp, 512]
-> latent-query compression [B, 256, 512]

cell id -> learned cell embedding

repeat N blocks:
  RNA = Cell-FiLM(RNA, cell)
  protein = Protein attends RNA
  RNA = RNA attends protein

final:
  RNA = RNA attends protein
  Z = RNA-centric representation [B, 600, 512]

outputs:
  Z -> profile head -> target/control/total profiles
  Z + target_prob -> gated binary pooling -> binding probability
```

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

## Ablations To Discuss

- `--protein-compression none`: full residue tokens instead of latent-query compression.
- `--protein-latent-len`: compare 128, 256, and 512 latent protein tokens.
- `--num-blocks`: compare 1, 2, and 3 fusion blocks.
- Cell conditioning variants: RNA-only FiLM, RNA+protein FiLM, or cell tokens in attention.
- Binary pooling variants: use only `target_prob`, only `binary_prob`, or the current gated mixture.
- Profile supervision variants: PureCLIP positives only, count threshold only, or binding-and-count.

## Modes

- `multimodal`: RNA + residue-level protein + cell.
- `rna-only`: zeros protein tokens and cell ids.
- `no-cell`: uses protein tokens but zeros cell ids.
- `protein-shuffle`: shuffles protein tokens across the batch.
