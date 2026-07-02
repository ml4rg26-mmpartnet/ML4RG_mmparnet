# Integration map — team-merge → main

This branch (`integration/team-merge`) merges all three workstreams so they coexist modularly and
expandably, ready for `main`. Nothing is removed; each feature lives behind a distinct module path.

## What's merged

| Source branch | Author | Contribution | Lands in |
|---|---|---|---|
| `dev/cgerards/early-fusion` | Christoph | M1 early-fusion baseline (concat[ProtT5, RNA]→MLP), notebook | `baselines/early_fusion.py`, `notebooks/early_fusion_baseline.ipynb` |
| `dgu/baseline-audit` | gudiyi | protein-conditioned dataset layer, ProtT5 track map, full-223 + 9-track PARNET eval, sanity scripts | `data/multimodal.py`, `scripts/{build_prott5_track_map,eval_*,sanity_*}.py`, `mmpartnet_out/prott5_track_map.tsv` |
| *ours (selected ×2)* | — | (A) **M2 nt-resolution profile conditioning**; (B) **conditioning mechanism + interpretability** | `models/{heads,early_fusion,cross_attn_head}.py`, `experiments/{m2_profile,binding_mechanism,xattn_faithfulness,binding_xattn_perres,...}.py`, demo nbs 10–11 |

We deliberately ported only our **two best, most-justifiable** contributions (calibrated to the current
team stage), each plugging into a colleague's branch:
- **(A) M2 profile conditioning** consumes the protein-conditioned sample framing of `data/multimodal.py`
  (window × track → profile). Result: conditioning ~doubles the per-nt profile Pearson, cross-cell robust.
- **(B) Mechanism + interpretability** extends `baselines/early_fusion.py` (the concat floor) into the
  controlled ladder concat → FiLM → cross-attention → per-residue, plus the model-internal interpretability
  (attention faithful to ISM; per-residue attention reads the RRM/KH domains).

## Modular layout (how they coexist)

```
src/mmpartnet/
  models/        parnet.py · heads.py (FiLM) · early_fusion.py (PARNET-feature concat) · cross_attn_head.py
  baselines/     early_fusion.py        # Christoph's standalone ProtT5 M1 baseline (kept distinct)
  data/          multimodal.py (gudiyi's conditioned sample layer) · sources/* · loader/registry
  experiments/   recover_demo_* · m2_profile · binding_{mechanism,xattn_perres,eval,head,grand} · xattn_faithfulness
  io/ process/ adapters/ ...           # shared foundation (from main)
scripts/         eval_* · sanity_* · build_prott5_track_map (gudiyi) · viz.py · nbgen.py (ours, notebook infra)
notebooks/demo/  00–03 (foundation) · early_fusion_baseline (Christoph) · 10–16 (ours) + executed/
mmpartnet_out/   committed result JSONs (recover_*, prott5_track_map, m2_profile*, binding_*, xattn_faithfulness)
```

Two `early_fusion` modules coexist on purpose: `baselines/early_fusion.py` is Christoph's standalone M1
ProtT5 baseline (trivial RNA encoder, by design); `models/early_fusion.py` is the PARNET-feature concat used
as the **floor of the mechanism ladder** in `binding_mechanism.py`. The ladder's concat cell *is* the
completed version of Christoph's "swap for PARNET body features" TODO.

## Our notebook suite (4 findings-grouped notebooks, all executed, on Moyon lab data)

Refined from 8 micro-notebooks into 4 grouped by the most significant findings. Each notebook is **thin**:
What/Why/Data → Definitions (LaTeX math) → `viz.fig_*(J("…json"))` builder → `display(Markdown)` result →
conclusion + leakage caveat. No hardcoded plotting — the figure logic lives in the reusable `scripts/viz.py`.

| nb | milestone | grouped findings |
|----|-----------|------------------|
| `10_m1_conditioning_and_fair` | **M1** | conditioning ladder + bidir-N2 objective + **the fair comparison** (per-residue is the only method that beats the RNA-only baseline; concat/FiLM lose; baseline ~½ leakage) |
| `11_interpretability` | **interp** | attention faithful to ISM (RNA side) + per-residue attention reads the RRM/KH domains (protein side) |
| `12_drivers_and_reliability` | M1 | protein-rep ablation (per-residue is the lever; ProtT5≈ESM; STRING not robust) + competence (trust = binding strength) + RNA-structure signal axis |
| `13_m2_profile` | **M2** | nt-resolution profile conditioned on protein (~2× profile, cross-cell HepG2/K562) |

### Reusable plotting (`scripts/viz.py`)
Schema-driven builders — `fig_fair`, `fig_mechanism`, `fig_faithfulness`, `fig_domains`, `fig_rep`,
`fig_competence`, `fig_structure`, `fig_m2` — each takes a result dict and adapts to whatever methods /
baselines it contains, composed from generic primitives (`violin_box`, `grouped_violins`, `delta_violins`,
`baseline_bars`, `col`). So a teammate's results JSON (same schema) re-plots with one call; nothing is
hardcoded per notebook.

**Private GPU job infrastructure is intentionally excluded** from this repo (it is our own). The notebooks
load committed result JSONs from `mmpartnet_out/` (computed on a CUDA GPU on the lab data) and re-plot —
reproducible from a clone without our orchestration. Hardware mentions are generic.

## Reconciliation notes
- **ProtT5 vs ESM:** the team standardized on ProtT5; we showed ESM≈ProtT5 (rep ablation), so either is fine.
- **`models/parnet.py`:** gudiyi's eval additions are kept; our lab-data `load_finetuned` helper (in the
  full repo) can be folded in next if the 9-track finetuned reference is wanted here.
- **deps:** `pixi.toml` h5py de-duplicated to the tighter pin; `ipykernel`/`pandas` (Christoph) + `h5py`/
  ViennaRNA-optional retained. `requirements.txt` mirrors.

## Expanding from here
- Wire `m2_profile` to consume `data/multimodal.py` directly (currently a thin internal loader) — one seam.
- Swap `ML4RG_PARNET_WEIGHTS` → leave-out PARNET to turn every in-distribution number into a clean
  held-out-RBP (zero-shot) claim; all harnesses re-run unchanged.

---

## Full-merge expansion (2026-07-01)

The conservative base above (cgerards + baseline-audit + our 2 best) is now expanded to fold in the
**remaining team branches** and our **new functionality**, all behind the modular seams. Full rationale
+ blob-level conflict topology in `docs/MERGE_PLAN.md`.

### Newly merged team branches
| Branch | Author | Lands in | Notes |
|---|---|---|---|
| `feature/rbp-binding-dataset` | — | `data/rbp_binding.py` (+ defensive `data/__init__`) | (RNA-window, RBP) binary dataset; optional import (h5py/lab mount) |
| `dgu/film-multitask` | gudiyi | `models/film.py` (canonical **multitask** FiLM), `experiments/film_multitask.py`, `protein/providers/prott5_h5.py` | registered `conditioning='film'` |
| `dgu/cross-attention` | gudiyi | `models/cross_attention_dgu.py` | registered `conditioning='xattn'` (A/B variant A) |
| `dfra/cross-attention-v2` | dfra | `models/cross_attention_dfra.py` | registered `conditioning='xattn2'` (A/B variant B) |

The two cross-attention variants **coexist** behind `models/registry.py` (one `conditioning` seam). The
A/B winner is deferred to a leave-out-RBP benchmark WITH controls on a GPU node; keep both until the
comparison notebook captures it, then drop the loser (MERGE_PLAN.md §8).

### New reusable modules (our functionality, lifted out of run-scripts)
| Module | Surface |
|---|---|
| `eval/` | `held_rbp_gate(score_fn,...)`, `run_controls`, `family_disjoint_assert`, `roc_auc`, `profile_pearson`, `partition_seen_unseen`, `CONTROLS`, `control_fired` (B3 fire-check) |
| `models/registry.py` | `list_heads` / `head_spec` / `build_head` — the plug-in seam for a teammate model |
| `metrics/coral.py` | `coral_f1_auroc(...,seen_mask)` (cold-start block), `validate_grid` (affinity far>near + family-block perm) |
| `cluster/mmseqs.py` | `load_clusters` / `cluster_fasta` / `n_families` — the correct family metric (~1744/99) |
| `scaling/family_curve.py` | `load_curve` / `slope` — the independent-family scaling curve loader |
| `adapters/coral.py` | `write_coral_csv` / `validate_roundtrip` — our data → CORAL schema |
| `splits/strategies/paralog.py` | leave-out-paralog (within-well transfer control) |
| `config.py` | new `RunConfig.conditioning` / `control`; `honest_zero_shot()` gate |

### The shared interface (so any teammate plugs in)
Drop a head in `models/`, implement `forward(rna_feats, protein_rep)`, add one `HeadSpec` row, select via
`RunConfig.conditioning`, and it runs through the SAME leakage-controlled, family-disjoint `eval/protocol`.
Worked demo (executed, synthetic, runs from a clone): `notebooks/demo/05_modular_eval_and_registry.ipynb`.

### Honesty gate (enforced, not documented-only)
`config.honest_zero_shot()` is **False** on the default leaked all-223 PARNET body, so no merge silently
upgrades a proxy number to a headline. Every zero-shot/multimodal claim stays proxy-level until
`ML4RG_PARNET_WEIGHTS` points at leave-out weights (+ `ML4RG_HONEST_ZEROSHOT=1`).

### Verification
`pytest src/mmpartnet/tests/test_modular_merge.py` — 7/7 pass (registry builds every head incl. both
cross-attn variants; eval controls fire-check; CORAL/affinity metrics; family split + clustering; adapter
round-trip). Leak-scan: the modules we authored carry no machine-specific paths; teammate files keep
their own VM paths (their code, faithfully merged).
