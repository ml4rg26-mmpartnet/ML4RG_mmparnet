# Cross-Attention Short Usage Note

This note explains how to run and inspect the cross-attention branch in a small,
practical way. It is written for the current `dgu/cross-attention` branch.


## Main Files

```text
src/mmpartnet/models/cross_attention.py
scripts/train_cross_attention_profile.py
scripts/eval_cross_attention_profile.py
scripts/export_cross_attention_interpretation.py
scripts/score_cross_attention_motifs.py
scripts/sanity_cross_attention_model.py
docs/CROSS_ATTENTION_WORKFLOW.md
docs/CROSS_ATTENTION_EXPERIMENT_STATUS.md
notebooks/demo/04_cross_attention_head_demo.ipynb
```

The model class is:

```python
ProteinCellCrossAttentionProfileHead
```

It takes frozen PARNET RNA features, residue-level ProtT5 protein embeddings, and a cell index, then predicts a binding profile and/or binary binding label.

## Setup

Clone the repository and switch to the cross-attention branch:

```bash
git clone https://github.com/ml4rg26-mmpartnet/ML4RG_mmparnet.git
cd ML4RG_mmparnet
git checkout dgu/cross-attention
```

Install PyTorch first. For GPU training, install the CUDA version matching the machine. This example uses CUDA 12.1:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies and this repo.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you want to open and rerun the demo notebook locally, install Jupyter and Matplotlib too:

```bash
python -m pip install jupyter matplotlib
```

## Quick Sanity Check

This command only checks whether the main Python files have valid syntax. It does not load data and does not train a model:

```bash
PYTHONPYCACHEPREFIX=/tmp/mmpartnet_pycache \
python -m py_compile \
  src/mmpartnet/models/cross_attention.py \
  scripts/train_cross_attention_profile.py \
  scripts/eval_cross_attention_profile.py \
  scripts/export_cross_attention_interpretation.py \
  scripts/sanity_cross_attention_model.py
```

This command runs a tiny fake batch through the cross-attention model. It checks that the forward pass, output shapes, profile distributions, binary outputs, and loss calculation are valid:

```bash
python scripts/sanity_cross_attention_model.py \
  --protein-len 120 \
  --protein-latent-len 64
```

This command does the same fake-batch sanity check, but with the new target-profile binary pooling mode:

```bash
python scripts/sanity_cross_attention_model.py \
  --protein-len 120 \
  --protein-latent-len 64 \
  --binary-alpha-source target
```

## Task Modes

The training script supports three task modes:

```text
--task multitask
  profile head + binary head

--task profile-only
  profile head only

--task binary-only
  binary head only
```

In the original multitask head, the binary pooling distribution is:

```text
alpha_bind = gate * target_prob + (1 - gate) * binary_position_prob
```

where:

- `target_prob` is the profile head's predicted target profile distribution.
- `binary_position_prob` is a separate position distribution learned from the binary objective.
- `gate` decides how much binary pooling uses the profile-derived distribution versus the binary-specific distribution.

Other available values are:

```text
--binary-alpha-source gated            # default, original behavior
--binary-alpha-source target           # use target_prob with binary gradient into profile branch
--binary-alpha-source target-detached  # use target_prob without binary gradient into profile branch
--binary-alpha-source binary           # use binary_position_prob only
```

For `binary-only`, the branch supports:

```text
--binary-pooling position  # learned position softmax, original binary-only behavior
--binary-pooling mean      # masked mean pooling
```

## Training Examples

These are example training commands. They can be copied and modified for larger or smaller runs. The exact training/evaluation interface may be updated after the shared main-branch infrastructure and collator are finalized.

The sanity script and demo notebook above use fake data, so they do not need any
dataset files. Real training does need the PARNET dataset, PureCLIP labels, ProtT5 embeddings, and the track-to-protein map.

On the project VM, the scripts already have default paths for those files. On a different machine, add these arguments to the training command and replace the example paths with wherever those files are stored locally:

```text
--hfds /home/dgu/storage_ml4rg26-shared/parnet-eclip/data-formatted-for-training/600nt_windows.no-one-hot.stripped/encode.filtered.hfds
--binding-dataset /home/dgu/storage_ml4rg26-mmparnet/manually_gathered/600nt_windows.no-one-hot.stripped.binding/600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt
--protein-h5 /home/dgu/storage_ml4rg26-mmparnet/manually_gathered/ProtT5_zenodo_datasets/embeddings_file.h5
--track-map mmpartnet_out/prott5_track_map.tsv
```

### Common Training Arguments

The commands below reuse the same core arguments. Their meanings are:

```text
--tracks all
```

An RBP-cell track is one eCLIP target track, usually one RBP in one cell line such as `QKI_HepG2` or `QKI_K562`. `--tracks all` uses all available tracks.

A comma-separated list such as `--tracks 9,138,195` uses only those track indices 9, 138, and 195. In the current track map, this example means `AQR_HepG2`, `QKI_HepG2`, and `U2AF2_HepG2`, which is useful for a smaller run.

```text
--max-train-windows 0
--max-valid-windows 0
```

Use all available RNA windows when building the train/validation sample pools. A positive number limits the number of RNA windows and is useful for small smoke tests.

```text
--valid-sample-size 32000
```

Evaluate on a fixed random validation subset of 32,000 flattened samples. This keeps validation cheaper than full validation while making runs comparable.

```text
--include-short
```

Include RNA windows shorter than 600 nt. These shorter RNA windows are padded to the model input length, and the valid-position mask tells the loss/evaluation which positions are real sequence.

```text
--batch-size 32
```

Number of flattened RNA-window/RBP-cell samples per batch.

```text
--epochs 5
--steps-per-epoch 1000
```

Train for 5 epochs. Each epoch runs 1000 training batches.

```text
--balanced-train
--balanced-pos-fraction 0.5
```

`--balanced-train` turns on balanced sampling for training batches. Without this flag, training batches are sampled from the dataset normally.

`--balanced-pos-fraction 0.5` sets the positive binding fraction after balanced sampling is turned on. `0.5` means each training batch has about 50% binding-positive examples and 50% binding-negative examples.

```text
--task multitask
```

Choose which heads are active. Options are `multitask`, `profile-only`, and `binary-only`.

```text
--lambda-profile 1
--lambda-binary 10
```

Weights for the profile loss and binary loss in the total objective. For
`profile-only`, `lambda_binary` is 0. For `binary-only`, `lambda_profile` is 0.

```text
--profile-mask-source binding
```

Choose which samples are allowed to train the profile head. `binding` means only PureCLIP-positive samples contribute profile loss. PureCLIP-negative samples can still train the binary head, but they do not train the profile head.

For `profile-only`, we often use `--balanced-pos-fraction 1.0` together with `--profile-mask-source binding`, because profile-only training gets useful loss only from binding-positive samples.

```text
--num-blocks 1
```

Number of cross-attention fusion blocks before the final RNA-attends-protein update.

```text
--protein-latent-len 256
```

Number of learned latent protein tokens after compressing residue-level ProtT5 embeddings.

```text
--binary-alpha-source gated
```

For `multitask`, choose how the final binary pooling distribution `alpha_bind` is built. `gated` is the original design:

```text
alpha_bind = gate * target_prob + (1 - gate) * binary_position_prob
```

`target` uses only the profile head's predicted `target_prob`:

```text
alpha_bind = target_prob
```

```text
--binary-pooling mean
```

For `binary-only`, choose how the binary head pools RNA positions. `position` uses a learned position softmax. `mean` uses masked mean pooling as a simpler alternative.

```text
--lr 3e-4
```

Learning rate for AdamW on the cross-attention head. The PARNET body is frozen.

```text
--device cuda
```

Train on GPU. Use `--device cpu` only for tiny sanity checks.

```text
--out-dir ./cross_attention_runs
--run-name RUN_NAME
```

Where to save checkpoints, logs, and `metrics.json`.

### Multitask, Original Gated Pooling

This command trains the original multitask cross-attention model. The binary head uses the gated mixture of `target_prob` and `binary_position_prob`:

```bash
cd ML4RG_mmparnet

python scripts/train_cross_attention_profile.py \
  --tracks all \
  --max-train-windows 0 \
  --max-valid-windows 0 \
  --valid-sample-size 32000 \
  --include-short \
  --batch-size 32 \
  --epochs 5 \
  --balanced-train \
  --balanced-pos-fraction 0.5 \
  --steps-per-epoch 1000 \
  --task multitask \
  --lambda-profile 1 \
  --lambda-binary 10 \
  --profile-mask-source binding \
  --num-blocks 1 \
  --protein-latent-len 256 \
  --binary-alpha-source gated \
  --lr 3e-4 \
  --device cuda \
  --out-dir ./cross_attention_runs \
  --run-name multitask_l10_lr3e4_latent256_5x1000_seed0
```

### Multitask, Target-Profile Binary Pooling

This command trains a multitask model where the binary head pools only with the profile head's predicted `target_prob`:

```bash
python scripts/train_cross_attention_profile.py \
  --tracks all \
  --max-train-windows 0 \
  --max-valid-windows 0 \
  --valid-sample-size 32000 \
  --include-short \
  --batch-size 32 \
  --epochs 5 \
  --balanced-train \
  --balanced-pos-fraction 0.5 \
  --steps-per-epoch 1000 \
  --task multitask \
  --lambda-profile 1 \
  --lambda-binary 10 \
  --profile-mask-source binding \
  --num-blocks 1 \
  --protein-latent-len 256 \
  --binary-alpha-source target \
  --lr 3e-4 \
  --device cuda \
  --out-dir ./cross_attention_runs \
  --run-name multitask_targetalpha_l10_lr3e4_latent256_5x1000_seed0
```

### Profile-Only

This command trains only the profile head. It is used to check how well the cross-attention backbone predicts the eCLIP profile without any binary loss:

```bash
python scripts/train_cross_attention_profile.py \
  --tracks all \
  --max-train-windows 0 \
  --max-valid-windows 0 \
  --valid-sample-size 32000 \
  --include-short \
  --batch-size 32 \
  --epochs 15 \
  --balanced-train \
  --balanced-pos-fraction 1.0 \
  --steps-per-epoch 500 \
  --task profile-only \
  --lambda-profile 1 \
  --lambda-binary 0 \
  --profile-mask-source binding \
  --num-blocks 1 \
  --protein-latent-len 256 \
  --lr 3e-4 \
  --device cuda \
  --out-dir ./cross_attention_runs \
  --run-name profile_only_lr3e4_latent256_15x500_seed0
```

### Binary-Only Mean Pooling

This command trains only the binary head with mean pooling. This is a simpler binary-only option than learned position-softmax pooling:

```bash
python scripts/train_cross_attention_profile.py \
  --tracks all \
  --max-train-windows 0 \
  --max-valid-windows 0 \
  --valid-sample-size 32000 \
  --include-short \
  --batch-size 32 \
  --epochs 5 \
  --balanced-train \
  --balanced-pos-fraction 0.5 \
  --steps-per-epoch 1000 \
  --task binary-only \
  --lambda-profile 0 \
  --lambda-binary 1 \
  --profile-mask-source binding \
  --num-blocks 1 \
  --protein-latent-len 256 \
  --binary-pooling mean \
  --lr 3e-4 \
  --device cuda \
  --out-dir ./cross_attention_runs \
  --run-name binary_only_mean_lam1_lr3e4_latent256_5x1000_seed0
```

To run the original binary-only comparison in the same style as the multitask run, use the learned position-softmax pooling and keep the binary loss weight at 10:

```text
--binary-pooling position
--lambda-binary 10
```

## Evaluation

This command evaluates a saved checkpoint on the fixed validation subset. Replace `RUN_NAME` with the folder name of the run to evaluate:

```bash
python scripts/eval_cross_attention_profile.py \
  --checkpoint ./cross_attention_runs/RUN_NAME/best_pearson.pt \
  --split valid \
  --max-windows 0 \
  --valid-sample-size 32000 \
  --batch-size 32 \
  --task multitask \
  --device cuda \
  --include-short \
  --out ./cross_attention_runs/RUN_NAME/eval_valid_32000.json
```

Useful ablation modes:

```text
--mode multimodal       # RNA + protein + cell
--mode no-cell          # RNA + protein + zero cell embedding
--mode rna-only         # RNA + zeroed protein embeddings + zero cell embedding
--mode protein-shuffle  # RNA + shuffled protein within batch + original cell
```

## Interpretation Export

This command exports per-sample predictions for interpretation. Replace
`RUN_NAME` with the folder name of the run to inspect:

```bash
python scripts/export_cross_attention_interpretation.py \
  --checkpoint ./cross_attention_runs/RUN_NAME/best_pearson.pt \
  --split valid \
  --max-samples 256 \
  --batch-size 8 \
  --task multitask \
  --out ./cross_attention_runs/RUN_NAME/interpretation_valid.pt
```

Important exported tensors include:

```text
target_prob
binary_position_prob
alpha_bind
binding_gate
binding_prob
pred_total
pred_control
mix_coeff
true eCLIP/control counts
binding_label
sequence and metadata
```
