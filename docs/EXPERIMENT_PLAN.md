# Family-diversity finding — rigorous experiment plan (2026-07-02)

Status: ACTIVE. Turns the famfull "diversity is the lever" signal into a properly-tuned, confound-gated,
per-approach mechanistically-interpretable program. Every result lands as an executed `.ipynb` (nbgen,
LaTeX math, plots, house style), every intermediary as reusable JSON in `mmpartnet_out/`, and model
weights are saved + manifested for reuse. Source: multi-lens design panel (mechanism / confounds / tuning
/ interpretability), 2026-07-02.

## Two flaws in the raw famfull 0.37 (must fix before any claim)
1. **Best-epoch-on-test peeking.** `scaling/family_curve.py::best_metric_from_val_csv` takes MAX MCC over
   epochs on the SAME held set that is reported (val==test). The mean-carrying folds PEAK AT EPOCH 0 then
   decay (fold1 0.589→0.432→0.366→0.385; fold3 0.376→0.338→0.174→0.201) — so it selects the *least-trained*
   checkpoint. Training on train-families HURTS held-family transfer = the OPPOSITE of interpolation.
2. **RNA not held out (confound, MEASURED).** Held RNAs are ~100% shared with train (pos-RNA overlap 68%);
   an RNA-only bindability baseline (protein ignored) already gets **MCC 0.234** vs CORAL 0.367. Most of the
   signal + most of famfull>famscale is RNA-coverage memorization. (`mmpartnet_out/famfull_confounds.json`)

## P0 — must run first (finding does not stand until these pass)
- **P0-1 De-peek + checkpoint.** Three-way FAMILY split per fold: TRAIN(~226) / EARLY-STOP-VAL(~60, family-
  disjoint) / TEST(held 60, never touched). Patch the CORAL fork's `scripts/train.py`: each epoch compute
  VAL-fold F1+MCC, `torch.save` best.pt on max VAL-F1 + always last.pt; `predict.py --checkpoint` emits
  per-pair `protein,rna,y,score`. Report per fold: (a) test@val-epoch [HONEST], (b) last-epoch, (c) best-on-
  test [PEEK]; (c)-(a)=peeking bias. Add `mcc()` to `eval/metrics.py`. ~9h (5×1.8h). → `S1_depeek_headline.ipynb`.
- **P0-2 Spuriousness gate (4 nulls, ALL must pass on de-peeked checkpoints; gate via
  `eval/controls.control_fired(min_gap=0.05,down)` + bootstrap CI):**
  1. **Protein-shuffle** (derange Protein_id→Prot_seqs across held RBPs, ≥5 seeds) → MCC must COLLAPSE.
  2. **RNA-memorization** → per-fold test RNA overlap; `partition_seen_unseen(name_key='RNA_id')` seen- vs
     unseen-RNA MCC; add `rna_disjoint_assert` + build ONE RNA-AND-family-disjoint fold. Require
     MCC(unseen-RNA) ≈ MCC(seen-RNA) + survival under RNA-disjoint.
  3. **RNA-only / random-body** (retrain with protein embedding zeroed) → interaction MCC must BEAT it.
  4. **Negative-sampling** (length/GC/k-mer-matched hard negatives, or negs from other-held-RBP positives)
     → MCC must survive. → `S2_null_battery.ipynb`, `S3_rna_overlap.ipynb`.
- **P0-3 Near-vs-far regression (mechanism: interpolation vs extrapolation).** Add
  `cluster/mmseqs.search_identity()` wrapping `mmseqs easy-search held_reps.fasta train_prot.fasta ...
  --format-output query,target,pident,qcov -c 0.8 -s 7.5`. Per held family `nearest_id` = max pident over
  train members (qcov≥0.8). Join to DE-PEEKED per-family MCC/AUROC (pooled ~300 fams). Fit MCC~nearest_id
  (Spearman+OLS), bootstrap CI, distance-shuffle null (1000×), far-bin (nearest_id<30%) vs chance. RNA-length
  covariate. → `F1_near_vs_far.ipynb`.

## Proper training protocol (no test-peek)
Three-way family split; early-stop on VAL-F1 (patience 3); save best.pt; reload; emit TEST preds ONCE.
Report mean±std over ≥3 model seeds × ≥5 negative-repair seeds. PRIMARY = AUROC (`eval/metrics.roc_auc`);
SECONDARY = MCC at a VAL-chosen threshold (never test-chosen); min_pos≥5/family. HP sweep on VAL ONLY:
lora_rank{8,16,32} × lr{1e-4,3e-4,1e-3} × batch{16,32} × epochs(early-stop). Two-stage: coarse (1 fold ×
18 configs, reduced epochs → top-2 by VAL-F1); confirm (top-2 × 5 folds × 3 seeds). Report winner's TEST
AUROC/MCC once + peeking-bias. Log `val_metrics.csv` + `test_at_val_epoch.csv`. Prefer the pooled ~300-family
per-family regression over the noisy 5-point N-curve (5 folds confound N with family identity). ~54 gpu-h
full confirm — batch across available GPU seats, coarse first.

## Per-approach matrix (keep CORAL interaction vs our PARNET heads STRICTLY separate)
| approach | proper-trained result | mechanistic readout | tool |
|---|---|---|---|
| CORAL interaction | family-disjoint-VAL early-stop, 5 folds×3×5 seeds → test@val-epoch AUROC+MCC | near-vs-far slope + identity-capped ablation + protein-shuffle attribution | `metrics/coral.validate_grid`, `coral_f1_auroc`, `cluster/mmseqs.search_identity` |
| conditioned (M1 binary) | rbp_holdout + `held_rbp_gate` | B3 control-fire + seen/unseen RBP | `eval/protocol.held_rbp_gate`, `partition_seen_unseen` |
| film (M2 profile) | leave-out-RBP zero-shot profile-Pearson | seen vs unseen Pearson + attention faithfulness | `eval/metrics.profile_pearson`, `xattn_faithfulness` |
| perres/perres_bidir (M2 nt-res) | leave-out-RBP nt-res Pearson | per-residue Pearson + attention faithfulness | `m2_decompose_zsdump`, `xattn_faithfulness` |
| xattn (dgu) / xattn2 (dfra) | arch ablation vs CORAL xattn, same protocol (AUROC) | attention attribution | per-fold predict |
→ `F5_per_approach_matrix.ipynb`.

## Notebook deliverables
S1_depeek_headline · S2_null_battery · S3_rna_overlap · F1_near_vs_far · F2_identity_capped_ablation ·
F3_density_partial_corr · F4_protein_shuffle_attribution · F5_per_approach_matrix · F6_density_matched_N (conditional).

## Reusability rules
- Weights: `best.pt`/`last.pt` per fold on the GPU host (per-fold output dir), + a manifest JSON in
  `mmpartnet_out/` pointing to them (config, VAL-epoch, seeds). Weights stay on the GPU host (large); the
  repo carries per-pair prediction dumps + metric JSONs.
- Intermediary results: JSON in `mmpartnet_out/` (famfull_confounds, per-fold preds summaries, near-far m8, etc.).
- Analysis logic: version-controlled `scripts/` in the repo; executed on the cluster, outputs fetched.
