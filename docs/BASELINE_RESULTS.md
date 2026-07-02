# Early-fusion baseline results

Run 2026-06-26 on the de.NBI VM (A4000). Sanity-check baseline for the multimodal
pipeline: simplest possible classifier that takes a 600-nt RNA window and an RBP
identity and predicts whether the RBP binds that window.

Purpose: validate that the data substrate + the gene→ProtT5 mapping + the
(window, RBP) label alignment are all correct, and that the protein channel can in
principle be used by a downstream multimodal model. **Not** a competitive number;
not a zero-shot claim.

## 1. Setup

| Piece | Value |
|---|---|
| Data substrate | `manually_gathered/600nt_windows.no-one-hot.stripped.binding.narrowpeak_intersect/dataset.pt` |
| Splits | predefined `train` 512,946 / `valid` 116,542 / `test` 70,626 |
| Protein rep | ProtT5 reduced (1024-d), `ProtT5_zenodo_datasets/reduced_embeddings_file.h5` |
| Symbol → vector map | `human.fasta` `GN=<symbol>` → sequential h5 index |
| RBP column source | `rbp_cts.tsv` (column-aligned to `binding` tensor) |
| RNA encoder | per-base mean of one-hot (4-d, position-agnostic) |
| Model | `concat[RNA(4), ProtT5(1024)] → MLP(256→256→1)` |
| Loss | `BCEWithLogitsLoss(pos_weight = (1-p)/p)`, `p` from a 5k sample |
| Optimizer | AdamW, lr=1e-4, batch=512, 3 epochs |
| Sampled training windows | 5,000 (of 512,946) |
| Sampled validation windows | 2,000 (of 116,542) |
| RBPs in subset | ~200 (after dropping those without a ProtT5 vector) |

Code: [src/mmpartnet/baselines/early_fusion.py](../src/mmpartnet/baselines/early_fusion.py),
runnable notebook: [notebooks/early_fusion_baseline.ipynb](../notebooks/early_fusion_baseline.ipynb).

## 2. Results

```
                     val_auroc    val_auprc
full model:            0.8276       0.0269
protein-shuffle:       0.5499       0.0101
RNA-only (zeroed):     0.6272       0.0101

Δ full − shuffle:     +0.2777
Δ full − RNA-only:    +0.2004
```

Both standing controls (protein-shuffle, RNA-only ablation) pass the agreed threshold
(Δ > 0.02 on each).

## 3. What this shows

- The data pipeline is correct end-to-end: `dataset.pt` records → RBP-column-aligned
  binding labels → ProtT5 vector lookup → trainable model.
- The protein channel is being used. The protein-shuffle drop from 0.83 → 0.55 is
  large; if the model had ignored the protein vector and learned only from RNA, the
  shuffle would have left AUROC unchanged.
- A trivial fusion (concat → MLP) clears chance by a wide margin on this data,
  validating that the multimodal task is at least learnable in the regime we have.

## 4. What this does NOT show

- **It is not a competitive number.** The RNA encoder is a 4-d base-frequency
  vector with no positional information. Most of the AUROC is plausibly driven by
  per-RBP global binding propensities (each RBP's ProtT5 vector encodes "what
  fraction of windows this RBP tends to bind") rather than learned RNA-protein
  interaction. A per-RBP frequency baseline (planned) will pressure-test this.
- **It is not zero-shot.** Train and validation share the same RBPs; only the
  windows differ. The clean zero-shot test requires the team's `splits.rbp_holdout`
  + a leave-out-pretrained PARNET, both lab-gated (see
  [DATA_INVENTORY.md](DATA_INVENTORY.md) §8).
- **AUPRC is small in absolute terms (0.027)** because positives are sparse
  (a few percent). AUROC is invariant to class balance and flatters the model
  under heavy imbalance. The classifier ranks well but does not retrieve precisely.

## 5. Caveats / known weaknesses

1. **Trivial RNA encoder** — base frequency over 600 nt, position-agnostic. Swap to
   PARNET body features (`models.parnet.load_parnet().body_feats(...)` mean-pooled
   to 512-d) is the obvious next step.
2. **Pooled AUROC only** — no per-RBP breakdown yet. 0.83 could mean "0.83 on every
   RBP" or "0.95 on PTBP1 / U2AF2, 0.50 on most others". The latter would change
   the interpretation substantially.
3. **Subsampled** — 5k of 512k train windows. Scale-up pending.
4. **Window splits, not chromosome splits** — adjacent windows from the same gene
   can land in train and valid. A chromosome-based split (held-out `chr1`, etc.)
   is stricter and pending.
5. **All controls run only at evaluation time.** The protein-shuffle is a
   post-training permutation, not a from-scratch retraining; this is the standard
   form of the control but it is the eval-time variant.

## 6. Reproduction

On the VM, in the cloned repo:

```bash
# environment (one-time)
export PIXI_CACHE_DIR="$HOME/pixi-cache"
pixi install
pixi add h5py pandas               # if missing
pixi run python -m ipykernel install --user --name mmpartnet --display-name "Python (mmpartnet)"

# data already at:
ls ~/storage_ml4rg26-mmparnet/manually_gathered/

# run the notebook end-to-end
# open notebooks/early_fusion_baseline.ipynb in VS Code,
# select kernel "Python (mmpartnet)", run all cells.
```

The notebook contains an `assert` after the first probe that fails loudly if the
`rbp_cts.tsv` row count does not match `binding.shape[0]` — i.e., if the column
alignment is silently wrong, the run aborts before training.

## 7. Next steps (ordered)

1. **Refactor** `EarlyFusionDataset` → model-agnostic `WindowRBPDataset` returning
   raw `{sequence, protein_vec, rbp, binding, meta}`. Encoding moves into each
   model. Unblocks Daniel (cross-attention) and Diyi (FiLM) on a shared loader.
2. **Per-RBP AUROC breakdown** — print top/bottom 5 and the distribution; verifies
   the pooled 0.83 is broad.
3. **Per-RBP frequency baseline** — score every window for RBP X as X's global
   positive rate. If early-fusion does not beat this by a large margin, the model
   is only learning priors.
4. **Swap RNA encoder to PARNET body features** (512-d mean-pooled). Expect
   RNA-only AUROC to rise, full to rise more, controls to still pass.
5. **Scale up** to 50k+ train windows once the per-RBP picture confirms the
   baseline is real.
6. **Chromosome-based split** for the test set to remove the within-gene leakage.

Items 1–3 are unblocked and can land this week. 4 depends on the lab's PARNET
weights being loadable in the pixi env (they are). 5 is mechanical. 6 needs a
small split helper added to `splits/strategies/`.
