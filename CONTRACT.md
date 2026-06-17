# Data contract + the 4 swap-in points

The whole repo is built around ONE internal data type and FOUR config-flag swap-ins,
so the harness runs on the substrate we have today and flips to the lab's canonical /
clean artifacts (when delivered) with a one-line change and zero downstream edits.

## The single cross-team interface: `ParnetDataElement`

`src/mmpartnet/contract.py` re-exports the lab's `ParnetDataElement` (from
`parnet_demo_utils`, import-or-vendor) plus `torch_sparse_to_dense` /
`torch_dense_to_sparse` / `make_element` / `validate_element`.

Every adapter emits, and every consumer reads, this one type:
- `inputs.sequence` — the (4, L) one-hot RNA window (L = 600 nt)
- `outputs["eCLIP"]` — the per-nt eCLIP target track (sparse)
- `outputs["control"]` — the RNA-only background track (PARNET's bias mixture)
- `meta` — RBP symbol, family, chrom, N1/N2 negative origin

**The protein stays OUT of the element** (it is per-RBP, not per-window): a parallel
`ProteinRep` (`io/embeddings.py`) keyed by RBP symbol, joined at batch assembly.

This signature is frozen first; everything else builds against it in parallel.

## The four swap-in points (all one-line config flags)

| # | What | Today (now) | Flips to (gated, lab) | Where |
|---|------|-------------|------------------------|-------|
| 1 | PARNET weights | leaked all-223 `NewRBPNet_7M` | a **leave-out-pretraining** PARNET | env `ML4RG_PARNET_WEIGHTS` |
| 2 | protein rep | `esm650_pooled` / `ribex_proxy` (ESM2+STRING) | `ribex_real` (lab-trained RIBEX) | `RunConfig.protein` |
| 3 | data substrate | `peaks` (public ENCODE eCLIP) | `hfds` (`encode.filtered.hfds`) | `RunConfig.substrate` → `adapters/hfds.py` |
| 4 | split axis | `naive` (held-chrom) / `family` | `rbp_holdout` (clean zero-shot) | `RunConfig.split` |

**Swap #1 is the decisive test.** `load_parnet()` and the entire harness are weight-agnostic, so
pointing `ML4RG_PARNET_WEIGHTS` at a leave-out-pretraining PARNET re-runs everything unchanged; on
held-out RBPs that is then an honest zero-shot evaluation (on the leaked all-223 weights it is not).

## Two standing controls (mandatory on any multimodal number)

When the protein-conditioning model is built, every multimodal number must be reported beside two
controls: (1) a **protein-permutation** control (shuffle the protein rep across RBPs; the score must
collapse, else the model is bypassing the protein), and (2) an **RNA-matched hard-N2** control
(negatives matched on the RNA window so only the protein differs; guards the RNA shortcut). Scope M1 as
tool/data-learning + sparsity handling, not a multimodal win, until both controls are in place.
