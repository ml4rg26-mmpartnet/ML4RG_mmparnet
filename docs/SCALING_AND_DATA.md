# Scaling, modular data layer, and the fair-eval fixes

Distilled from a multi-agent adversarial review of the methods + eval and a read of the actual data layer.

## 1. Modular data layer — CONSOLIDATE onto what already ships (not greenfield)

The merge already contains `data/{loader,base,registry,preprocess,multimodal}.py`, `data/sources/*`,
`protein/registry.py` + `protein/providers/*`, and `splits/{base,registry,strategies/*}`. The job is to make
every experiment consume ONE `DataConfig`-driven loader instead of the current ad-hoc loaders.

**Target contract** — `DataConfig`: `source` (binding_pt | hfds | encode_bigwig | local_pt), `task`
(binding | profile), `scheme`, `cell`, `split` (naive | family | rbp_holdout) + `split_config`,
`protein_reps` (esm | prott5 | perres | string | combos), `seq_len`, `group`, `n_rbps`, `target_kind`,
`batch_size`, `extra`. `UnifiedDataLoader(cfg)` → dataset + collator returning
`{sequence, rbp, protein:{rep→array}, binding:(B,T) | eclip+control:(B,L), window_meta}`.

**Duplications to retire** (found across the repo):
- `load_split()` reused by binding_head/grand/fair; `feats_pos()` in grand+fair; mean+max pooling in 3+ files.
- 3 protein joins: early_fusion (h5 + GN→idx), binding_fair/grand (direct .npz), m2_profile (different npz keys).
- `sparse_track_to_dense` in multimodal.py + m2_profile.py; per-residue pad+mask in both.
- RBP↔track indexing reimplemented per experiment; the `iter_elements` flat-Dataset split bug.

**Migration order** (each step keeps everything working; assert metric parity before deleting old code):
1. Finish `protein/registry.py` as the single lookup (fold the 4 npz loads from `io/embeddings.py`; add
   `prott5_reduced.npz`; lazy per-residue). Kills early_fusion's h5 path + the direct .npz loads.
2. `data/sources/binding_pt.py` (`@register("binding_pt")`): load `{scheme}/dataset.pt`, flatten to
   (window,rbp) pairs like `multimodal.py`, implement `rbps/windows/sequence/observed`. Replaces `load_split`.
3. Finish `data/sources/hfds.py`: `_build_track_map()` absorbs m2_profile's `_cell_select/_to_dense`; move
   `sparse_track_to_dense` to `preprocess.py`.
4. Flesh out `data/loader.py:UnifiedDataLoader`: source→split (existing `splits/registry`)→protein_reg→dataset
   →collator. Move `feats_pos` + pooling into the collator; per-residue pad+mask into a shared ProfileCollator.
5. Migrate, in order, with parity asserts: `binding_head` → `binding_grand` → `binding_fair` (also apply the
   fair-fixes here) → `m2_profile` → `baselines/early_fusion`.
6. Delete the ad-hoc loaders; add `docs/DATA_LAYER.md`; wire `tests/test_sanity.py` to run each experiment
   headless as a parity integration test.

`data/multimodal.py` (gudiyi) is the spine for the conditioned (window×track→profile) sample; keep it.

## 2. Scale-up roadmap

- **Panel 68 → 223.** ESM2 covers 223/223; STRING-PE ~80-85% (fine for film/xattn); **per-residue is the
  bottleneck (68/223)**. Phase 1a: run film/xattn on the full 223 ESM panel now (`binding_grand panel=full`
  flag exists). Phase 1b (blocker for per-residue at scale): compute the missing ~155 ESM2-650M per-residue
  embeddings (fair-esm, optional extra), reduce (L,1280)→(L,32), extend the perres npz with padding masks.
- **Data 25k → 100k → 500k.** Phase 1: 100k windows, both cells (HepG2+K562), in-memory (~13 GB feats, fits
  32 GB). Phase 3 (conditional): full hfds ~500k → won't fit GPU → pre-compute PARNET body_feats once to HDF5,
  batch-stream (~4 ms/batch SSD, overlapped). Frozen features are cached once per run (the key efficiency).
- **Mock zero-shot now (cheap honest proxy):** cell-holdout — train HepG2, eval K562 (both already loaded by
  m2_profile). One boolean flag. Tests cross-context generalization while the leave-out checkpoint is pending.
- **Leave-out PARNET (the decisive test):** infra ready (`m2/leaveout_parnet.py`); deploy = set
  `ML4RG_PARNET_WEIGHTS` + `ML4RG_PARNET_HELDOUT`, `split="rbp_holdout"`; <5 min, harness unchanged.
- **Phase-1 analysis gate:** only go to Phase 3 if the conditioning gap vs RNA-only GROWS with scale.
- **Expected effect:** in-distribution gap stays modest (projected +0.02-0.04 auPRC, ceiling ~0.13-0.14 vs
  RNA-only ~0.10-0.11). The story flips to "defensible contribution" only if the method keeps a significant
  gap-vs-shuffle under real leave-out zero-shot — plan the narrative around that inflection, not the
  in-distribution number.

## 3. Fair-eval fixes (apply before supervisors — `binding_fair.py`)
Adversarial review (3 lenses, all "not sound") found:
- **P0 leakage labelling:** runs on leaked all-223 PARNET but printed as "FAIR". → stamp body-training status
  into print + JSON; gate on a leave-out checkpoint or `--allow-leaked`; re-title "in-distribution".
- **P0 statistics:** the promised paired sign test is not implemented; `15/68` is actually significant
  *negative* evidence (binomial p~1e-5). → add `binomtest` + Wilcoxon signed-rank on per-RBP paired deltas,
  print p-value + direction.
- **P0 bootstrap:** non-paired + fixed seed → artefactually tight CIs (features frozen across seeds correlate
  them). → paired bootstrap with a live RNG; optional seed-level (hierarchical) resample.
- **P1 equal regime:** one lr (5e-4) for all; same epochs (drop the baseline's +3); one loss policy.
- **P1 architecture parity:** the RNA-only baseline is a multitask head (cross-RBP sharing) vs per-RBP
  conditioned heads → add a per-RBP single-task RNA-only baseline; report deltas vs both.
- **P1 missing controls:** run `parnet_zeroshot` (frozen floor) + a random-body RNA-only baseline to quantify
  how much of the 0.107 baseline is leakage vs real RBP structure (report all three).
- **P2:** per-residue 32-dim vs others 1280-dim (footnote or upgrade); add BH/FDR across methods×RBPs; report
  auROC alongside auPRC; relabel `rna_only_bind` → "protein-agnostic branch" (it's from the protein-trained head).
