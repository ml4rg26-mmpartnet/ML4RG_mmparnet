# Integrated merge — map, verification, and push guide

Branch `integration/team-merge` unifies all three workstreams. This is the map to verify it and push it
forward.

## 1. What is in the merge

```
src/mmpartnet/
  models/      parnet · heads (FiLM) · cross_attn_head (xattn + per-residue) · early_fusion (PARNET-feat concat) · ribex
  baselines/   early_fusion.py            # Christoph's standalone ProtT5 M1 baseline
  data/        loader · base · registry · preprocess · multimodal (gudiyi) · sources/{hfds,local_pt,encode_bigwig,...}
  protein/     registry · providers/*     # the one protein-rep lookup (ESM/ProtT5/per-residue/STRING)
  splits/      base · registry · strategies/*   # naive / family / rbp_holdout
  experiments/ recover_demo_* · binding_{head,eval,grand,fair,mechanism,xattn_perres} · m2_profile · binding_n2_xattn · binding_ribex · binding_structure · binding_competence
  io/ process/ adapters/ ...              # shared foundation
scripts/        eval_* · sanity_* · build_prott5_track_map (gudiyi) · viz.py (REUSABLE schema-driven plot builders) · nbgen.py
notebooks/demo/ 00-03 (foundation) · early_fusion_baseline (Christoph) · 10-13 (ours, findings-grouped) + executed/
mmpartnet_out/  committed result JSONs   docs/  INTEGRATION · PERFORMANCE_SUMMARY · MERGE_MAP · SCALING_AND_DATA
```

Per-team: **Christoph** = `baselines/early_fusion.py` + nb. **gudiyi** = `data/multimodal.py`, `scripts/eval_*`,
ProtT5 track map. **Ours** = `models/{heads,cross_attn_head,early_fusion}`, the binding/M2 experiments,
notebooks 10-17, viz/nbgen, the controls (`binding_grand`: derangement + within-family). Private GPU job
infra is intentionally excluded.

## 2. Status of each piece
- **Coexists + imports**: verified (our modules import; colleague h5py modules need the declared `h5py` dep).
- **Notebooks 10-17**: executed, demo format, math + violin graphs, on Moyon lab data.
- **Data layer**: the modular scaffolding is PRESENT (`data/loader`, `protein/registry`, `splits/*`) but the
  experiments still use ad-hoc loaders (`binding_head.load_split`, per-file `feats_pos`, 3 ProtT5 joins).
  Consolidation plan in `SCALING_AND_DATA.md` — this is the main code-debt to retire.
- **`binding_fair.py` (notebook 17)**: NEEDS the fairness/stats fixes below BEFORE supervisors (adversarial
  review flagged it: labelled "FAIR" while on leaked all-223 PARNET, multitask-vs-per-RBP baseline, unequal
  lr/epochs/loss, sign test not implemented, bootstrap artefactually tight). Fixes are cheap and in progress.

## 3. The honest caveat that must travel with the numbers
All in-distribution results use the **leaked all-223 PARNET** (body trained on the test RBPs). Adversarial
review showed the **RNA-only baseline (0.107) is itself leakage-inflated** — so the methods look *worse* than
they are, while the protein-shuffle gaps (+0.06-0.08) prove protein IS useful. The **decisive test is the
leave-out PARNET** (env swap, harness unchanged). Frame every supervisor-facing number as "in-distribution;
leave-out pending".

## 4. Verify checklist (do not trust self-report — execute)
1. **Build**: `pixi install` (or `pip install -r requirements.txt`) → `python -c "import mmpartnet"`.
2. **Sanity/parity**: `python -m pytest src/mmpartnet/tests/test_sanity.py` headless (CI green).
3. **Notebooks**: open `notebooks/demo/executed/10..13_*_executed.ipynb` — figures + numbers present. Each is
   thin: it calls a reusable `scripts/viz.py` `fig_*` builder on a `mmpartnet_out/*.json` (no hardcoded
   plotting), so the same notebook re-plots any teammate's same-schema results. Numbers pulled via f-strings.
4. **Fair-eval fixes fired**: open `mmpartnet_out/binding_fair.json` and confirm it records body-training
   status, a real sign-test/Wilcoxon p-value, equalized lr/epochs/loss, and the zero-shot-floor + random-body
   controls (see §3 of `SCALING_AND_DATA.md`). Re-run `python -m mmpartnet.experiments.binding_fair pureclip 5`
   and diff the JSON.
5. **Data-loader parity** (when consolidation lands): each migrated experiment reproduces its pre-refactor
   metric within tolerance; keep the old loader on a side branch until parity is signed off.

## 5. Push sequence
> ⚠ **Branch hygiene.** The current working branch is `fix/foam-nlm-connector-scope-downgrade` with unrelated
> foam-nlm changes staged. Do **not** push the merge from it. Cut a clean branch off `main` and bring only the
> `dist/mmparnet-merge` content.

1. **Leak-scan / content guard**: the GitHub repo is the PUBLIC `ml4rg26-mmparnet/ML4RG_mmparnet`. Confirm
   `.gitignore`/`.gitattributes` exclude the all-223 weights, real PARNET checkpoints, `hfds`, ProtT5/ESM npz,
   and any Paperspace-gated/patient data. `git add -A --dry-run` must show no data/weights/`*.npz`/`*.pt`.
2. **Commit in honest order**: (a) the fairness-fixes commit on its own (so reviewers see the correction in
   isolation), (b) the data-loader consolidation, (c) the rest.
3. **PR** against the team repo with the binding_fair before/after JSON + the parity table in the body.
4. **Tell supervisors (Moyon/Bernecker)**: in-distribution numbers are leakage-confounded; the leave-out-PARNET
   run is the pending decisive test (we need the checkpoint + held-RBP manifest).

## 6. The one thing that flips the story
A modest in-distribution gap (per-residue +0.014 vs the inflated RNA-only baseline) becomes a **defensible
contribution only under leave-out PARNET zero-shot**. Everything in the merge is built to re-run unchanged on
that weight swap — that is the experiment to prioritize with the supervisors. See `SCALING_AND_DATA.md`.
