# Consolidated Merge Plan — MultiModal PARNET (mmpartnet)

Status: DRAFT for review (do not auto-commit; lczamprogno commits). Generated 2026-07-01.
Scope: fold **our new functionality** (leakage battery, family-scaling, CORAL verification,
decomposition, interpret, M1/M2 methods) into the git-tracked consolidated version
(`integration/team-merge`) AND merge the six teammate branches cleanly + reusably.

---

## 0. The three trees (ground truth)

| Tree | Path | What it is |
|---|---|---|
| Team clone | `dist/ML4RG_mmparnet` (`main` @ c8f5475) | pristine team `main` + remote-tracking branches |
| **Integration worktree** | `dist/mmparnet-merge` (`integration/team-merge` @ 1c17e59) | **the consolidated version we build here**; already has our 3 commits (PARNET eval + ProtT5 map + dataset sanity) + an *early* cgerards merge |
| Our dev package | `poc/ml4rg_parnet/mmpartnet` (monorepo, uncommitted) | our full working package — 90 files, notebooks, nbgen tooling; source of the **new functionality** |

Remote branches (fetched 2026-07-01): `main`, `dev/cgerards/early-fusion`, `dgu/baseline-audit`,
`dgu/cross-attention`, `dgu/film-multitask`, `dfra/cross-attention-v2`, `feature/rbp-binding-dataset`.

---

## 1. Divergence manifest (poc dev vs integration branch)

- **39 files only in our dev** = the new functionality to bring in (all of `eval`/leakage,
  `m2_decompose`, `binding_gate`, interpret, M1/M2 experiments, `train/select/losses/analysis/diagnostics`,
  `io/synthetic`, tests). Additive — no collision.
- **3 files only in integration** (keep) = cgerards `baselines/early_fusion.py`, `baselines/__init__.py`,
  `data/multimodal.py`.
- **20 files in both but different** = reconcile (poc wins for our-owned files; UNION for the shared spine
  `config.py`, `models/parnet.py`, `models/__init__.py`).
- **47 identical**.

## 2. Blob topology of the hot shared files (across branches)

| File | Distinct blobs | Canonical decision |
|---|---|---|
| `models/film.py` | 2 — `ef74` (cgerards + dgu-film, **206 ln, multitask `task=` switch**) vs `6e96` (audit + xattn + dfra, 182 ln, stripped) | **`ef74` canonical** (multitask superset); register `conditioning='film'` |
| `models/cross_attention.py` | 2 — `994b` (dgu) vs `6cf5` (dfra) | **genuine A/B**; land BOTH behind `xattn_variant`; benchmark → winner default |
| `data/multimodal.py` | 3 — cgerards/film (222) · audit (197) · **xattn/dfra (229)** | **xattn/dfra `2a49` (229 ln) = superset**, take it |
| `models/parnet.py` | main(77) → team-common(112) → **dfra(115)** ; **ours(91)** diverged separately | **UNION**: team ProtT5-track-map eval + our `.body_feats()`/loader |
| `config.py` | main(=most) vs `3d6ec` (xattn/dfra add xattn keys) ; **ours** diverged | **UNION**: our swap-point config + xattn keys + new `conditioning`/`control[]` keys |
| `models/__init__.py` | 5 (per-branch registrations) | **UNION** all head registrations |
| `protein/providers/prott5_h5.py` | 1 (identical cgerards=dgu-film) | take as-is |
| `experiments/film_multitask.py` | 2 (cgerards 6f63 · **dgu-film 4785**) | dgu-film owner → take dgu-film |
| `data/rbp_binding.py` | 1 (feature branch) | clean additive, take as-is |

## 3. Branch dispositions

| Branch | Action | Unique feature files to take | Notes |
|---|---|---|---|
| `feature/rbp-binding-dataset` | **merge FIRST** | `data/rbp_binding.py` (+ `data/__init__` union) | pure-additive data layer; unblocks the RBP-binding substrate |
| `dgu/baseline-audit` | **merge early** | sanity/eval scripts (`sanity_*`, `eval_223track_*`, `eval_9track_*`), `film_profile_sanity_summary.json` | audit findings → fold into `eval/controls.py`; no model conflict |
| `dev/cgerards/early-fusion` | **merge** | `models/early_fusion.py`, `baselines/early_fusion.py`, `protein/providers/prott5_h5.py`, `docs/BASELINE_RESULTS.md`, `scripts/{build_prott5_track_map,eval_film_multitask,train_film_profile}.py`, `notebooks/early_fusion_baseline.ipynb`, `mmpartnet_out/prott5_track_map.tsv` | register `conditioning='early'` |
| `dgu/film-multitask` | **merge (owns FiLM+multitask)** | `models/film.py` (`ef74` canonical), `experiments/film_multitask.py` (`4785`), `docs/FILM_MULTITASK_WORKFLOW.md` | register `conditioning='film'` |
| `dgu/cross-attention` | **merge → A/B** | `models/cross_attention.py` (`994b`, variant `dgu`), `docs/CROSS_ATTENTION_*.md`, xattn scripts | consumes `perres64.npz` |
| `dfra/cross-attention-v2` | **merge → A/B** | `models/cross_attention.py` (`6cf5`, variant `dfra_v2`), residue-level ProtT5 xattn | head-to-head vs dgu on leave-out-RBP + controls |
| **OUR dev** | **merge-modularize** | 39 new files + 20 reconciled → lift into `eval/`, `scaling/`, `cluster/`, `metrics/coral.py`, `adapters/coral.py`, `models/registry.py`, `splits/strategies/{chromosome,paralog}.py` | see §5 |

## 4. Merge order (conflict-avoiding)

0. Snapshot integration branch (tag) before touching it.
1. **feature/rbp-binding-dataset** → `data/rbp_binding.py` (+ `data/__init__` union). Additive.
2. **dgu/baseline-audit** → sanity/eval scripts + audit → `eval/controls.py`. No model files.
3. **Port OUR new functionality** (39 additive files) into the tree.
4. **Reconcile the 20 diverged** (poc-wins for our-owned) + author the 3 UNION spine files
   (`config.py`, `models/parnet.py`, `models/__init__.py`).
5. **Modularize** — lift logic into `eval/`, `scaling/`, `cluster/`, `metrics/coral.py`,
   `adapters/coral.py`, `models/registry.py`, split strategies (§5).
6. **cgerards early-fusion** → `models/early_fusion.py` + `prott5_h5` provider + docs; register `early`.
7. **FiLM** → canonical `ef74` `models/film.py` + `film_multitask.py`; register `film`.
8. **Cross-attention A/B LAST** → both `cross_attention.py` variants behind `xattn_variant`
   in `models/registry.py`; benchmark on `rbp_holdout` + protein-permutation/RNA-matched controls;
   winner = default, loser kept only until the comparison notebook is captured.
9. Freeze `models/registry.py` (`film|xattn|xattn2|early`), run tests, tag `v0.1`.

## 5. Modularization (lift OUR logic out of `experiments/` into importable subpkgs)

| New module | Lifts from | Public surface |
|---|---|---|
| `eval/controls.py` | `experiments/{eval_controls,binding_gate}.py` | `CONTROLS = {name: fn(model,batch,split)->delta}`: protein_shuffle, rna_matched_neg, family_mean_floor, paralog_floor, shuffled/circular/center_bump nulls. A control that fails to move its metric is flagged (B3 discipline). |
| `eval/decompose.py` | `experiments/m2_decompose.py` | `decompose(model, splits) -> {in_dist, zero_shot} x {binary, profile, affinity}` |
| `eval/metrics.py` | `eval.py` | profile-Pearson, binary auROC/auPRC, seen/unseen partition |
| `eval/protocol.py` | `m2/generalization.py` | `run(model, split, controls) -> {seen, unseen, control_deltas}`; **guarantees family-disjoint eval** (zero-family-overlap assertion) |
| `scaling/family_curve.py` | famscale run scripts | `family_scaling(train_family_counts, eval='heldout') -> curve` |
| `cluster/mmseqs.py` | mmseqs pipeline | `cluster_families(fasta) -> {protein: family_id}` (1744 families / 99 labeled) |
| `metrics/coral.py` | `coral_metric*.py`, `validate_grid.py` | `coral_f1_auroc(pred,y,groups) -> {overall,seen,unseen}`, `validate_grid(...)` (affinity far>near + family-block permutation) |
| `adapters/coral.py` | `scripts/eclip_to_coral{,_transcript}.py` | `eclip_to_coral(...)`, `eclip_to_coral_transcript(...)` + `validate_roundtrip()` |
| `models/registry.py` | (new) | `name -> ConditioningHead`; `film|xattn|xattn2|early` |
| `splits/strategies/chromosome.py` | `experiments/binding_cv_chrom.py` | leave-out-chromosome |
| `splits/strategies/paralog.py` | paralog-floor logic | paralog-block hold-out |

## 6. Shared interfaces (so ANY teammate model plugs into leakage-controlled, family-disjoint eval)

1. **Data contract** — `contract.ParnetDataElement` (frozen; RNA one-hot (4,600) + eCLIP/control counts). Unchanged.
2. **Config contract** — `config.RunConfig` with the 4 swap points (substrate/protein/split/model+loss)
   + new `conditioning: film|xattn|xattn2|early` and `control: list[str]`, + assertion
   `honest_zero_shot=False` unless `ML4RG_PARNET_WEIGHTS` points at leave-out weights.
3. **Model contract** — `ConditioningHead` protocol (`models/registry.py`):
   `forward(rna_feats:(B,512,L), protein_rep) -> ProfileOutput` with `binding_logit/binding_prob/profile`
   + `loss_components()`. FiLM, cross-attn (dgu), cross-attn-v2 (dfra), early-fusion register by name.
   **Diyi's model = drop a file in `models/`, implement the protocol, register a name.**
4. **Eval contract** — `eval/protocol.py::run(model, split, controls)` enforces family-disjoint eval +
   runs the mandatory Control registry, flagging any control that does not move the metric.

## 7. Notebooks capturing our recent findings (reusable by Diyi)

| Notebook | Captures | Reuses |
|---|---|---|
| `notebooks/leakage/L1_decomposition.ipynb` | in-dist protein signal (+0.2 auROC) is **RBP-identity lookup**; clean leave-out-RBP zero-shot ~null@binary, +0.03@profile, real+growing@affinity | `eval/decompose.py`, `eval/controls.py`, `models/registry` |
| `notebooks/leakage/L2_controls_battery.ipynb` | full control battery in one pass; each control MUST move its metric | `eval/controls.py`, split strategies |
| `notebooks/scaling/S1_family_scaling_curve.ipynb` | interpolation test: metric vs #independent families (N=10..200), vs FLAT paralog/window scaling | `scaling/family_curve.py`, `cluster/mmseqs.py`, `metrics/coral.py` |
| `notebooks/coral/C1_coral_reproduction.ipynb` | component-wise = true cold-start; released ckpt LEAKS (F1 0.92 vs 0.65 paper); clean per-fold retrain ~0.57 | `adapters/coral.py`, `metrics/coral.py` |
| `notebooks/coral/C2_eclip_to_coral_converter.ipynb` | runnable Moyon eCLIP+affinity → CORAL-format + round-trip validation | `adapters/coral.py` |
| `notebooks/scaling/S2_affinity_validate_grid.ipynb` | affinity is real+growing: far>near + family-block permutation | `metrics/coral.py::validate_grid` |
| `notebooks/aggregate/interpolation_narrative.ipynb` | CAPSTONE reframe: protein-conditioned binding = interpolation in RBP-space; ~4 wells can't, 99–1744 can | outputs of L1/S1/C1 |

## 8. Gates / risks (do NOT let a merge silently upgrade a proxy result to a headline)

1. **Zero-shot / multimodal CLAIMS stay proxy-level** until leave-out-pretrained PARNET weights +
   real RIBEX arrive. `honest_zero_shot=False` gate in `RunConfig`.
2. **Cross-attn A/B may be inconclusive on the leaked all-223 body** — label numbers proxy-level;
   run A/B WITH controls on `rbp_holdout`.
3. **mmseqs family count (1744/99)** underpins the scaling curve — pin the clustering config in
   `cluster/mmseqs.py` and snapshot the family map.
4. **Take-ours discipline on the shared spine** (`contract.py`, `config.py`, `parnet.py`, `io/embeddings.py`) —
   cherry-pick only each branch's UNIQUE feature files; author unions for the 3 spine files. Avoids 3-way churn.
5. **No commits by the assistant** — everything staged for lczamprogno to review (`git diff`) + commit.
6. **Private GPU job / node infrastructure stays OUT** of this repo (node names, jump host, seat-probe,
   cluster run-scripts). Results carry **generic hardware provenance only** ("a CUDA GPU"), never node names.

## 9. Consolidation baseline + re-sync (as of 2026-07-02)

This branch is consolidated at the following remote HEADs. All 7 branches folded in; the two `dgu/*` that
advanced were re-folded (a per-file refresh, not a re-merge — the `models/registry.py` seam keeps heads wired).

| branch | consolidated @ | notes |
|---|---|---|
| `main` | `c8f5475` | trunk |
| `dev/cgerards/early-fusion` | `a8ad380` | `models/early_fusion.py`, `prott5_h5` provider |
| `dfra/cross-attention-v2` | `871d589` | `models/cross_attention_dfra.py` (variant `xattn2`) |
| `dgu/baseline-audit` | `f87adb2` | audit -> `eval/controls` |
| `dgu/cross-attention` | `e2ccdd2` | `models/cross_attention_dgu.py` (variant `xattn`) + scripts |
| `dgu/film-multitask` | `d4e271a` | canonical FiLM (`models/film.py`) + `film_multitask` |
| `feature/rbp-binding-dataset` | `e9ab8be` | `data/rbp_binding.py` |

**Re-sync when a teammate pushes (light):**
```
git -C <clone> fetch --all --prune
# compare each origin/<branch> HEAD to the table above; for any that ADVANCED:
git -C <clone> show origin/<branch>:src/mmpartnet/models/<their_head>.py > src/mmpartnet/models/<their_head>.py
# (+ any updated scripts/docs). The registry already points at the file -> no re-merge.
pytest src/mmpartnet/tests/test_modular_merge.py     # confirm the seam still holds
```
You never edit their repos; you only refresh the changed feature files into this consolidated branch.
