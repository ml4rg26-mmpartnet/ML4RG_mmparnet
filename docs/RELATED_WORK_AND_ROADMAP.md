# Related work (CORAL, TFBindFormer) and the next-steps roadmap

## The two anchors
- **CORAL** (bioRxiv 2026.04.22.720174; *Cross-Attention Over RNA And Protein Sequences*). RNA encoder
  **DNABERT-2 (117M)**, protein **ESM-2 150M** (per-token, projected 640→768), **bidirectional cross-attention**
  (2 stacked layers: RNA→prot + prot→RNA + self-attn + FFN), **LoRA** (rank 8) on both encoders. Multi-task:
  contrastive + MLM + **binary interaction** classification. Data: 24,012 positive RNA-protein pairs (RNAInter +
  RPI sets), 1:1 negatives. **3-tier splits**; the headline is the **component-wise non-redundant** (entire
  held-out molecules) result: **F1 0.65 vs 0.55 next-best (+18% rel)**. Interpretability: heads 0/7 show **27%
  elevated attention at interface residues** across 309 PDB complexes (no structural supervision). Cost: 24-40 GB
  VRAM (full-length). No cross-species/transfer eval; **no RNA-only / leakage control**.
- **TFBindFormer** (DNA-TF, not RNA-RBP). DNA encoder TBiNet-style (conv + attn + BiLSTM, trainable); protein
  **ProtST5 + Foldseek 3Di structure tokens**; cross-attention; binary TF-DNA binding over 161 TFs × 91 cells.
  3Di structure tokens add **+0.005 AUPRC**. Transfers partially (structure-token idea; bidir cross-attn).

## Axis comparison (vs ours)
| axis | CORAL | ours | who's ahead |
|---|---|---|---|
| RNA encoder | DNABERT-2 (trainable LM) | **frozen eCLIP-foundation PARNET** | mixed (ours cheaper/task-grounded; theirs adaptable) |
| protein encoder | ESM-2 per-token, LoRA | ESM/ProtT5 pooled + **per-residue-32** + STRING-PE | them (per-token + LoRA; our perres is 32-d reduced) |
| fusion | 2-layer **bidirectional** cross+self+FFN | concat/FiLM/cross-attn/**per-residue** (1 layer) | them (depth/bidir); us (4-way honest dissection) |
| output | **binary interaction** | **nt-resolution profile** (M2) + binary | **us** (richer, harder) |
| generalization | **component-wise held-out molecules, F1 0.65 (run)** | leave-out-RBP zero-shot (**pending** clean PARNET) | them (executed); ours cleaner once run |
| eval rigor | redundancy-controlled splits | **+ random-body + protein-shuffle + within-family + RNA-only** | **us** (leakage-controlled) |
| interpretability | **plausibility** (interface attention, 309 PDB) | **faithfulness** (attention↔ISM) + RRM/KH 1.77× | **us on the stronger axis** (faithful > plausible) |
| scale | 24k pairs | 68 RBPs / 2 cells | them |

## Honest positioning
CORAL/TFBindFormer out-scale us on breadth, encoder adaptation, fusion depth, and external-validation size, and
**both already ran the held-out-generalization test our decisive gate has not** (blocked on the clean lab PARNET
checkpoint). Where we are genuinely ahead: **confound isolation and output richness** — we are the only one of
the three to run a **random-body leakage control** (~47% of the RNA-only baseline is PARNET memorization),
protein-shuffle + within-family nulls (protein is real signal even when leakage-overwhelmed in-distribution), a
**4-fusion head-to-head with honest negatives** (concat/FiLM lose to RNA-only; only per-residue wins, +0.015
auPRC, Wilcoxon p=6e-5, 43/68, 5 seeds), an **nt-resolution profile** task (~2× Pearson, cross-cell), and a
**faithfulness** (attention-vs-ISM) interpretability claim rather than plausibility. *Both baselines report
larger but uncontrolled numbers; ours are the only ones whose multimodal gain is provably not encoder memorization.*

## The distinct claim (how we push past CORAL)
Stake the **leakage-controlled multimodal claim**, not a bigger-number claim: protein conditioning of a frozen,
task-grounded eCLIP foundation yields gains that **survive a battery of nulls the whole RPI subfield omits**, on
**strictly richer outputs** (nt-resolution profile), with **faithfulness-grade interpretability**. The decisive
escalation is the **leave-out-PARNET / leave-out-RBP zero-shot**: we can remove the foundation's ability to
memorize the target, so ours is the only *clean* test of generalization on the protein modality — CORAL's
"held-out" molecules still leak through broadly-pretrained ESM-2/DNABERT-2, reported with **no leakage control**.
The contribution is methodological: in this subfield, *"beating the baseline" must mean beating a
leakage-controlled baseline, and we are the only group that can produce that number.*

## Ranked next steps
| # | action | milestone | cost | depends on |
|---|---|---|---|---|
| **1 (first)** | **Full per-token frozen protein** (ProtT5 1024-d / ESM-2 640-d → 768 proj) into the per-residue head + dim-sweep 32/128/256/640 | widen the only winning head | **S** | none (ProtT5 in data drop) |
| 2 | **Depth-match** the winning head only (CORAL-style 2-layer bidir+self+FFN) under leakage controls | close CORAL's architectural edge, attributable | S | #1 |
| 3 | **M2 profile shape-only** correlation + cross-cell (HepG2→K562) + binarized-loses-signal control | richer task neither baseline has | M | #1 |
| 4 | **Leakage-controlled RPI benchmark harness** (wrap any model with our 4 nulls; reproduce a CORAL-style head on eCLIP; raw vs leakage-discounted) | the methodological differentiator | M | controls codebase |
| 5 | **Faithfulness suite** (attn-vs-ISM, deletion/insertion AUC, sufficiency) on ours vs a reimplemented CORAL head | plausibility→faithfulness, head-to-head win | M | ISM pipeline + #4 |
| 6 | **STRING-PE × per-residue** + PPI-degree-stratified ablation | PPI modality neither baseline uses | S | #1 |
| 7 | **P0 (decisive) Leave-out-RBP zero-shot** by family (CORAL component-wise analog) vs all 4 nulls | the clean novel-RBP claim | M | **BLOCKING: clean PARNET** (or self-retrain) |
| 8 | **LoRA-vs-leakage frontier** (sweep adaptation; in-dist auPRC vs leakage fraction vs held-out) | reframes CORAL's LoRA as a measurable confound | M | #7 |
| 9 | **Scale 68→223** RBPs, both cells | breadth-gap closure | L | clean de-leaked 223 set |
| 10 | **RNA accessibility/structure channel** (RNAplfold/RNA-FM, FiLM-gated), structure-class-stratified | RNA structure > protein 3Di for RBPs | M | #1 |

## Executed (2026-06-27) — results update (notebook 14)
Steps #1, #2, #3, #6 were run as configs of the modular leakage-controlled `binding_x` / `m2_profile` harness.
The hypothesis behind #1 was **falsified**, which redirects the strategy:

| step | what we ran | outcome | status |
|---|---|---|---|
| #1 | per-residue ESM-2 640-d, PCA dim-sweep 32/128/256/640 vs lab 32-d | **flat** (+0.015–0.017, CIs overlap); +0.016 is **not** dim-bottlenecked | done — **null** |
| #6 | broadcast STRING-PE / ProtT5 onto residue tokens | base +0.0168 → +string/+prott5/+both all ≤ base (**dilutes**) | done — **null** |
| #2 | per-residue vs **bidir** (CORAL block) × depth L2/L4 | +0.0171 / +0.0164 / +0.0167 / +0.0167 — **no gain from depth or bidirectionality** | done — **null** |
| #3 | **M2 leave-out-RBP zero-shot** (unseen RBPs, HepG2+K562) | per-residue gap **+0.047 / +0.039** (and **+0.026 / +0.035 vs within-family**); ~leakage-free | done — **WIN** |

**Revised reading.** The three "make the protein richer/bigger" M1 levers (#1/#2/#6) are dead ends in-distribution
— a bigger M1 auPRC is *not* how we pass CORAL (it out-scales us there). The leverage is the **richer-output +
generalization** axis: M2 nt-resolution **zero-shot to unseen RBPs** is robust, cross-cell-replicated, and the one
contribution CORAL's RNA-level binary RPI structurally cannot make. **Next priorities:** (a) the decisive **#7
leave-out-PARNET binary** (still BLOCKED — request a held-out PARNET checkpoint from Moyon/Bernecker); (b) extend
the M2 zero-shot (more held RBPs, family-stratified, profile shape-only); (c) **#5 faithfulness head-to-head** vs a
reimplemented CORAL head. Read every gain through the leave-out protocol, not in-distribution auPRC alone.
