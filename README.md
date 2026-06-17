# MultiModal PARNET (mmparnet)

**ML4RG SS26, Project 07** (TUM, Marsico lab; supervisors Lambert Moyon and Tobias Bernecker).

Condition the lab's frozen **PARNET** (RNA sequence -> per-nucleotide eCLIP crosslink profile, 223
RBP x cell tracks) on a **protein representation** (**RIBEX** = ESM2-650 + STRING-PPI personalized-PageRank
PE + FiLM) to predict RBP binding, with the goal of **generalizing to RBPs without eCLIP data**. Following
Moyon's steer, the contribution is the loss and the signal, not the architecture; the emphasis is on
**interpretability** and **honest competence bounds**.

This repository is the runnable substrate plus the tests that verify our public-data proxies behave as
intended. Start with **[QUICKSTART.md](QUICKSTART.md)**.

## Two caveats (read before any number)

1. **Leaked PARNET.** The public `NewRBPNet_7M` frozen body has seen every ENCODE RBP, so family-/RBP-holdout
   numbers are NOT clean zero-shot. The clean test awaits a leave-out-pretraining PARNET (see
   [CONTRACT.md](CONTRACT.md), swap-in #1). The naive held-chromosome split is the clean in-distribution result.
2. **Proxy RIBEX.** The protein representation is an ESM2-650 + STRING-PE proxy, not the lab's trained RIBEX;
   absolute numbers shift when RIBEX lands (`protein='ribex_real'`).

Any future multimodal number must be reported beside two standing controls (a protein-permutation gate +
an RNA-matched hard-N2); see [CONTRACT.md](CONTRACT.md).

## Quickstart

```bash
pip install -e ".[eclip,viz,interp]"      # or: pixi install
bash scripts/fetch_all.sh                  # public ENCODE peaks + hg38 -> data/  (weights: ask a supervisor)
bash scripts/run_demos.sh                  # execute the demo notebooks -> notebooks/demo/executed/
```

Full setup (venv, the cu128 nightly torch for RTX 5090, gated weights) is in **[QUICKSTART.md](QUICKSTART.md)**.

## Demo notebooks (`notebooks/demo/`)

Each states what is tested / why / the reasoning, gives the math definitions, runs a thin call into
`src/mmpartnet`, and pulls its numbers live. Pre-run copies (with outputs) are in `notebooks/demo/executed/`.

| notebook | demonstrates / tests | needs |
|---|---|---|
| `00_mmpartnet_demo` | modular data layer -> frozen PARNET -> profile recovery; switching source/format | weights + network |
| `01_proxy_validity_controls` | density proxy vs established nulls (center-confounded) vs established 5' crosslink counts (beats nulls) | weights + network |
| `02_finetune_negative_controls` | head-finetune vs pretrained, with random-body + faithful-RBPNet-objective controls | weights + network |
| `03_interpretability_mixcoeff` | PARNET's per-RBP additive mix-coefficient (sequence- vs bias-driven) | committed JSON only |

Headline results (executed on an RTX 5090): density recovery is center-confounded (Pearson +0.211, loses to
a center-bump), while the established crosslink-count target beats every null (+0.289); the head-finetune
gain is +0.089 and survives the random-body control (-0.043) but is loss-sensitive under the faithful
objective (+0.012).

## Layout

```
src/mmpartnet/    importable package: data/ protein/ splits/ m2/ (swappable layers) + models/ experiments/ io/ adapters/ process/
notebooks/demo/   4 runnable demo notebooks (+ executed/ copies with outputs)
mmpartnet_out/    precomputed result JSONs (committed; notebooks render on a fresh clone)
docs/             DATA_INVENTORY.md (have / surrogate / missing) + diagrams
CONTRACT.md       (repo root) the data type + the 4 swap-in points
scripts/          fetch_all.sh, run_demos.sh, fetch_data.sh, build_embeddings.py
metadata/         public ENCODE metadata (eclip manifest, cohort, RBP list) used by the fetch
data/             external assets (gitignored; populated by scripts/fetch_all.sh)
```

The package is built around one data contract + four one-line config swap-ins (PARNET weights, protein rep,
data substrate, split axis); see [CONTRACT.md](CONTRACT.md). Everything path-related is
environment-overridable, so a fresh clone runs zero-edit once the data is fetched.

## Citations

PARNET / RBPNet (Horlacher et al., Genome Biology 2023), RIBEX (Firmani et al., bioRxiv 2026), eCLIP / ENCODE
(Van Nostrand et al., Nat Methods 2016), ATtRACT, ESM2 (Lin et al., Science 2023), STRING.

## License

MIT (see [LICENSE](LICENSE)).
