# Performance summary — basic signal → M1 → M2

All numbers computed on the Moyon/Marsico lab canonical data (frozen PARNET `parnet.7m-0.0`, lab binding
datasets, full-223 `encode.filtered.hfds`, ESM/ProtT5/per-residue, ATtRACT domains). **All results are
in-distribution**; the leave-out-PARNET swap (`ML4RG_PARNET_WEIGHTS`) turns each into a clean held-out-RBP
(zero-shot) claim with the harness unchanged. Figures referenced live in `notebooks/demo/executed/`.

## Basic signal — the substrate works
| check | result | source |
|---|---|---|
| frozen PARNET profile recovery (5' crosslink counts) | Pearson **+0.289** vs circular-null −0.015, center-bump +0.23 | nb 01 (`recover_demo_profile_counts.json`) |
| frozen PARNET zero-shot binding (RNA only) | auPRC 0.029, lift **1.3×** | `binding_eval_pureclip.json` |
| trained RNA-only multitask head | auPRC 0.082, lift **5.3×** | `binding_head_pureclip.json` |

## Milestone 1 — conditioning mechanism, objective, interpretability
| result | number | figure |
|---|---|---|
| conditioning ladder (binding auPRC gain vs protein-shuffle) | concat +0.040 < FiLM +0.066 < cross-attn +0.081 < per-residue +0.082 | `nb10_mechanism.png` |
| objective (N2 protein-discriminative gate) | bidir-N2 **+0.136** vs BCE +0.016; FiLM wins N2 (0.686) | `nb10_mechanism.png` |
| protein representation | per-residue is the lever; ProtT5 ≈ ESM; STRING not robust | `nb13_ribex.png` |
| competence / trust signal | binding strength Spearman **+0.79**; OOD distance −0.14 | `nb14_competence.png` |
| interpretability (RNA side) | attention ↔ ISM **+0.30** real vs +0.16 shuffled (55/64 windows) | `nb11_faithfulness.png` |
| interpretability (protein side) | attention in RRM/KH domains **1.77×** (10/10 RBPs), unsupervised | `nb12_domains.png` |
| RNA-structure signal axis | accessibility adds +0.0023 auPRC (CI excludes 0) | `nb15_structure.png` |

## Milestone 2 — nt-resolution profile conditioned on protein (started)
| cell | best head | profile Pearson real vs shuffle | gap | within-family |
|---|---|---|---|---|
| HepG2 | per-residue | 0.214 vs 0.104 | **+0.110** | +0.08 (~73% specific-protein) |
| K562 | per-residue | 0.225 vs 0.123 | **+0.102** | +0.10 (~95% specific-protein) |

→ Conditioning on the correct protein **~doubles** the per-nucleotide profile correlation, cross-cell robust;
the gain is mostly specific-protein (survives a within-family shuffle). Figure `nb16_m2_profile.png`.

## Fair comparison vs an RNA-only baseline (hardened; notebook 17)
One harness, common panel (K=68), equal lr/epochs, paired bootstrap + binomial/Wilcoxon sign test, plus a
random-body control. RNA-only multitask baseline = **0.106**; **random-body = 0.057** → ~47% of the baseline
is PARNET-leakage. Methods vs RNA-only:

| method | vs RNA-only | sign test | verdict |
|---|---|---|---|
| concat (early-fusion) | −0.021 (CI<0) | 5/68, p=8e-14 | significantly loses |
| FiLM | −0.002 (CI<0) | 22/68, p=5e-3 | marginally below |
| cross-attn | +0.010 (CI>0) | 35/68, p=0.90 | beats on mean, not per-RBP |
| **per-residue** | **+0.015 (CI>0)** | **43/68, Wilcoxon p=6e-5** | **beats RNA-only (mean & per-RBP)** |

All methods beat the protein-shuffle null (specificity holds). In-distribution the RNA-only baseline is
leakage-inflated; the decisive test is the same harness under leave-out PARNET.

## Scaling the conditioned head toward CORAL (notebook 14)
Three M1 architecture/representation levers, each a config of the one leakage-controlled `binding_x` harness
(K=68, RNA-only baseline 0.106, random-body 0.057). **All null in-distribution:**

| lever | configs | gap vs RNA-only | verdict |
|---|---|---|---|
| **P1** widen per-residue protein | PCA dim 32 / 128 / 256 / 640 (vs lab 32-d) | +0.0165 / +0.0172 / +0.0168 / +0.0151 | flat — the +0.016 gain is **not** dim-bottlenecked |
| **P2** add global protein context | base / +STRING / +ProtT5 / +both | +0.0168 / +0.0148 / +0.0158 / +0.0147 | broadcast context **dilutes** (mildly hurts) |
| **P3** capacity / bidirectionality | perres L2 / L4 · bidir L2 / L4 | +0.0171 / +0.0164 / +0.0167 / +0.0167 | depth & CORAL's bidir block add **nothing** |

→ Richer/bigger protein conditioning does not move the in-distribution M1 number. Cheaply rules out the easy
directions and redirects effort to the axes below.

## M2 zero-shot — leave-out-RBP nt-profile generalization (notebook 14, the win)
`M2_SPLIT=rbp`: 30% of RBPs held out entirely; one protein-conditioned head predicts an **unseen** RBP's per-nt
profile from its protein rep. Far less PARNET-leakage-confounded than the binary task (the profile head is
trained leave-out; PARNET is only a frozen feature extractor).

| cell | head | profile Pearson real vs shuffle | gap vs shuffle | gap vs within-family |
|---|---|---|---|---|
| HepG2 | per-residue | 0.156 vs 0.109 | **+0.047** | **+0.026** |
| K562  | per-residue | 0.164 vs 0.125 | **+0.039** | **+0.035** |
| HepG2 | FiLM | 0.140 vs 0.084 | +0.056 | +0.027 |
| K562  | FiLM | 0.106 vs 0.091 | +0.015 | +0.013 |

→ The **per-residue** head generalizes robustly across both cells to RBPs never seen in training, and uses
RBP-specific (not just family-level) protein info; FiLM is cell-variable. **This is the contribution CORAL's
RNA-level binary RPI cannot make** (no nt-resolution profile). `nb14_m2_zeroshot.png`.

## Open gap (flagged)
The decisive in-distribution-leakage killer is the **leave-out-PARNET binary** test (same harness, swap
`ML4RG_PARNET_WEIGHTS`), where the leaked RNA-only baseline collapses on held-out RBPs and the conditioned
advantage should widen. It is **blocked**: the lab `parnet.7m-0.0` is trained on all 223 tracks (leaked); a
PARNET checkpoint trained *without* the held-out RBPs is needed — worth requesting from Moyon/Bernecker. The
M2 zero-shot above is the partial, ~leakage-free version of this claim that we *can* run today.
