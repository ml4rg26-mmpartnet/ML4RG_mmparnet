"""Build the recent-findings notebooks (executed) in the shared thin/rigorous style: ONE question ->
definitions -> load committed result JSON (computed on a GPU node) -> display -> conclusion + leakage
caveat. Results are loaded from mmpartnet_out/ (the committed shared space). Where a number needs the
CORAL repo / a GPU to recompute, the notebook DISPLAYS the recorded GPU result and states how it
was computed (see each JSON's `provenance`).

  python scripts/make_nb_findings.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbgen import md, code, build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ── L1: leakage decomposition ────────────────────────────────────────────────────────────────────
L1 = [
    md("# L1 - Decomposition: what protein-conditioning REALLY buys\n"
       "**Question.** Is the protein signal biology (transferable to unseen RBPs) or an RBP-IDENTITY "
       "lookup (memorized, non-transferable)? We decompose it by regime.\n\n"
       "*Data: `mmpartnet_out/{m2_profile,m2_profile_zeroshot_hepg2,binding_fair}.json` (computed on a "
       "CUDA GPU on the Moyon lab data; re-plotted here from a clone).*"),
    md("## Definitions\n"
       "- **in-distribution** real Pearson: train + eval on all RBPs (protein can act as an identity key).\n"
       "- **protein-shuffle**: derange protein reps across RBPs; a collapse => the signal was identity lookup.\n"
       "- **zero-shot** real Pearson: leave-out-RBP split (the RBP was never in head training).\n"
       "- **leakage-attributable AUPRC**: gap between the RNA-only baseline on the real PARNET body vs a "
       "random-init body = the share of the 'baseline' that is PARNET pretraining leakage."),
    code("import json; from pathlib import Path; from IPython.display import Markdown, display\n"
         "from mmpartnet.eval.decompose import regime_table\n"
         "OUT = Path('..') / '..' / 'mmpartnet_out'\n"
         "def J(n): return json.loads((OUT/n).read_text(encoding='utf-8'))\n"
         "mp, mz, bf = J('m2_profile.json'), J('m2_profile_zeroshot_hepg2.json'), J('binding_fair.json')\n"
         "rows = []\n"
         "for arch in ('film','perres'):\n"
         "    idd, zs = mp['archs'][arch], mz['archs'][arch]\n"
         "    rows.append((arch, idd['real'], idd['shuf'], idd['real']-idd['shuf'], zs['real'], zs['shuf']))\n"
         "leak = bf['leakage_attributable_auprc']\n"
         "reg = regime_table(profile={'in_dist': mp['archs']['perres']['real'],\n"
         "                            'zero_shot': mz['archs']['perres']['real'],\n"
         "                            'shuffle':  mz['archs']['perres']['shuf']})\n"
         "print('regime (profile axis):', reg['profile']['verdict'])"),
    code("tbl = '| arch | in-dist real | in-dist shuffle | shuffle collapse | zero-shot real | zero-shot shuffle |\\n'\n"
         "tbl += '|---|---|---|---|---|---|\\n'\n"
         "for a, ir, ish, gap, zr, zsh in rows:\n"
         "    tbl += f'| {a} | {ir:.3f} | {ish:.3f} | {gap:+.3f} | {zr:.3f} | {zsh:.3f} |\\n'\n"
         "display(Markdown(tbl))\n"
         "display(Markdown(f'**Leakage-attributable AUPRC** (RNA-only real body - random body): "
         "`{leak:.3f}` -- part of the baseline itself is PARNET pretraining leakage.'))"),
    md("## Conclusion\n"
       "In-distribution the protein signal is large but the **protein-shuffle collapses it** => it was an "
       "RBP-identity lookup, not biology. On the leave-out-RBP (zero-shot) split the real Pearson is "
       "small-but-positive (per-residue > FiLM), i.e. a *real* residual signal survives once identity is "
       "removed. **Leakage caveat:** the frozen body is the leaked all-223 checkpoint, so even the "
       "zero-shot number is PROXY-level (`config.honest_zero_shot()` is False) until leave-out PARNET "
       "weights are set. The binary axis zero-shot is ~null on a clean backbone (see the M1 leave-out-RBP "
       "gate, computed on a CUDA GPU clean-backbone run); the profile + affinity axes are where signal lives."),
]

# ── S1: independent-family scaling (motivated by the null feature/capacity levers) ─────────────────
S1 = [
    md("# S1 - Independent-family scaling is the lever (not capacity)\n"
       "**Question.** In-distribution feature/capacity levers were NULL. Does zero-shot transfer instead "
       "rise with the number of INDEPENDENT RBP families seen? (The interpolation-in-RBP-space reframe: "
       "~4 wells cannot interpolate; ~99-1744 mmseqs families can.)\n\n"
       "*Data: `mmpartnet_out/binding_x_p{1,2,3}.json` (committed); family curve computed on a CUDA GPU, "
       "results committed under `mmpartnet_out/famscale/` (see provenance in the last cell).*"),
    md("## Definitions\n"
       "- **P1 dim-sweep** (widen per-residue token dim), **P2 STRING-fusion** (+PPI PE), **P3 capacity** "
       "(depth) -- all measured vs the RNA-only baseline AUPRC (gap_vs_rna_only).\n"
       "- **family curve**: train on N independent mmseqs families, eval HELD-OUT families; sweep N."),
    code("import json; from pathlib import Path; from IPython.display import Markdown, display\n"
         "OUT = Path('..') / '..' / 'mmpartnet_out'\n"
         "def J(n): return json.loads((OUT/n).read_text(encoding='utf-8'))\n"
         "tbl = '| lever | method | gap vs RNA-only AUPRC |\\n|---|---|---|\\n'\n"
         "for f, lev in [('binding_x_p1_dimsweep.json','P1 widen tokens'),\n"
         "               ('binding_x_p2_stringfusion.json','P2 +STRING'),\n"
         "               ('binding_x_p3_capacity.json','P3 depth')]:\n"
         "    d = J(f)\n"
         "    for mname, mv in d['methods'].items():\n"
         "        g = mv.get('gap_vs_rna_only')\n"
         "        if g is not None: tbl += f'| {lev} | {mname} | {g:+.3f} |\\n'\n"
         "display(Markdown(tbl))\n"
         "display(Markdown('All levers hover near 0 (in-distribution) -> capacity/features are NOT the lever.'))"),
    code("# family-scaling curve from the committed GPU results (mmpartnet_out/famscale/famscale_N/0/)\n"
         "import csv, os\n"
         "from pathlib import Path as _P\n"
         "FAM = _P(os.environ.get('ML4RG_FAMSCALE', OUT / 'famscale'))\n"
         "def _read(N):\n"
         "    f = FAM / f'famscale_{N}' / '0' / 'val_metrics.csv'\n"
         "    if not f.exists(): return None\n"
         "    rows = [r for r in csv.DictReader(open(f)) if r.get('MCC') not in (None,'')]\n"
         "    if not rows: return None\n"
         "    acc = [float(r['Accuracy']) for r in rows]; mcc = [float(r['MCC']) for r in rows]\n"
         "    return {'nEp': len(rows), 'bestAcc': max(acc), 'bestMCC': max(mcc), 'lastMCC': mcc[-1]}\n"
         "tbl = '| N families | epochs | best Acc | best MCC | last-epoch MCC |\\n|---|---|---|---|---|\\n'\n"
         "for N in (10,25,50,100,200):\n"
         "    r = _read(N)\n"
         "    tbl += (f'| {N} | {r[\"nEp\"]} | {r[\"bestAcc\"]:.3f} | {r[\"bestMCC\"]:.3f} | {r[\"lastMCC\"]:+.3f} |\\n'\n"
         "            if r else f'| {N} | - | (pending) | | |\\n')\n"
         "display(Markdown(tbl))\n"
         "display(Markdown('_Metric = MCC / Accuracy on the FIXED held-out families. F1 is omitted: it is "
         "degenerate here (peaks at epoch 0, i.e. before learning). Near-chance across N (Acc ~0.51-0.54, "
         "MCC ~0.02-0.10), NO clean monotonic rise; last-epoch MCC ~0 or negative._'))"),
    code("# DEFINITIVE test: famfull = ALL 346 families, 5 disjoint held-out folds (mmpartnet_out/famfull/)\n"
         "import csv as _csv, statistics as _st\n"
         "FF = OUT / 'famfull'\n"
         "bestM = []; lastM = []; bestA = []\n"
         "tab = '| fold | best MCC (epoch) | last MCC | best Acc |\\n|---|---|---|---|\\n'\n"
         "for k in range(5):\n"
         "    f = FF / str(k) / 'val_metrics.csv'\n"
         "    if not f.exists(): continue\n"
         "    rows = list(_csv.DictReader(open(f)))\n"
         "    m = [float(r['MCC']) for r in rows]; a = [float(r['Accuracy']) for r in rows]\n"
         "    bi = m.index(max(m)); tab += f'| {k} | {max(m):.3f} (ep{bi}) | {m[-1]:+.3f} | {max(a):.3f} |\\n'\n"
         "    bestM.append(max(m)); lastM.append(m[-1]); bestA.append(max(a))\n"
         "display(Markdown(tab))\n"
         "if bestM:\n"
         "    display(Markdown(f'**FAMFULL (346 families, 5-fold): best-epoch MCC = {_st.mean(bestM):.3f} +/- "
         "{_st.pstdev(bestM):.3f} (Acc {_st.mean(bestA):.3f}); last-epoch MCC = {_st.mean(lastM):.3f} +/- "
         "{_st.pstdev(lastM):.3f}.** vs famscale N<=200 near-chance (MCC 0.02-0.10) -> diversity IS the lever.'))"),
    md("## Conclusion -- the null is OVERTURNED at maximal diversity\n"
       "This curve is the **CORAL architecture** (ESM-2-150M + DNABERT2 + bidirectional cross-attention + LoRA) "
       "-- the diversity-enabled RNA-protein INTERACTION model, no PARNET body (no all-223 leakage here).\n\n"
       "The famscale curve (N=10..200) was near-chance, which read as a null. But the DEFINITIVE test -- "
       "training on **ALL 346 families** (the full CORAL+Moyon pool), **5 disjoint held-out folds** -- "
       "generalizes to held-out FAMILIES: **best-epoch MCC 0.37 +/- 0.16 (Acc 0.68); last-epoch MCC "
       "0.29 +/- 0.12**, far above the N<=200 near-chance. So the flat low-N curve was **UNDER-COVERAGE of "
       "RBP-space, not a ceiling**: **independent-family diversity IS the lever** -- exactly the interpolation-"
       "in-RBP-space prediction (sparse wells cannot interpolate; dense coverage can).\n\n"
       "Honest reading:\n"
       "1. **Metric** -- best-epoch peeks at the test; last-epoch (0.29) is the conservative number and still "
       "wins decisively. Folds 1/3 peak at epoch 0 then overfit training families -> a proper early-stop on a "
       "TRAIN-family val would land ~0.3-0.4.\n"
       "2. **Mechanism (to verify, not hand-wave)** -- holding out 60 of 406 families leaves near-relatives in "
       "training, so this is likely dense-NEIGHBOUR interpolation (which IS the hypothesis) rather than transfer "
       "to isolated families. NEXT: near-vs-far distance-stratified check (does held-family MCC decay with "
       "sequence distance to the nearest training family?).\n"
       "3. **Reconciles C1** -- CORAL's component-wise ~0.57 (paralog-permissive) vs family-held-out 0.29-0.37 "
       "(paralog-excluded, harder) now sit on ONE axis: performance scales with how densely RBP-space is "
       "covered near the held-out point.\n\n"
       "### !! CONFOUND (see S2) -- this is NOT yet clean protein-family transfer\n"
       "The held **RNAs are ~100% shared** with training (only proteins are held out), and an **RNA-only "
       "bindability baseline (protein ignored) reaches MCC ~0.23** vs CORAL's 0.37 -- so **most of this signal, "
       "and most of the famfull>famscale gain, is RNA-coverage memorization**, not protein-family "
       "interpolation (more families -> more training RNAs -> held RNAs better covered). The genuine "
       "protein-family residual is ~0.13 MCC (best-epoch), highly fold-variable. **Treat 'diversity is the "
       "lever' as CONFOUNDED pending the clean RNA-disjoint + family-disjoint split + protein-shuffle "
       "control.** See `notebooks/scaling/S3_rna_overlap`.\n\n"
       "**Provenance:** CORAL fork trained on a CUDA GPU; per-fold results staged in `mmpartnet_out/famfull/` "
       "(RNAs truncated to 1000nt, batch 8, 4 epochs/fold, 5 disjoint 60-family held-out sets). famscale "
       "curve: `mmpartnet_out/famscale/`."),
]

# ── C1: CORAL direct verification ────────────────────────────────────────────────────────────────
C1 = [
    md("# C1 - CORAL direct verification (the cold-start existence proof)\n"
       "**Question.** What does CORAL actually achieve on a TRUE cold-start, and is the headline number "
       "clean? We ran the CORAL repo directly (not a re-implementation).\n\n"
       "*Data: `mmpartnet_out/coral_reproduction.json` -- recorded from a CUDA GPU run; see its "
       "`provenance` (repo, env, protocol). Displayed here because recompute needs the CORAL repo + a GPU.*"),
    md("## Definitions\n"
       "- **component-wise split**: 0% protein AND 0% RNA overlap with train = the true cold-start.\n"
       "- **Path A**: honest per-fold retrain (train only on that fold's train.csv)."),
    code("import json; from pathlib import Path; from IPython.display import Markdown, display\n"
         "d = json.loads((Path('..')/'..'/'mmpartnet_out'/'coral_reproduction.json').read_text(encoding='utf-8'))\n"
         "p = d['provenance']\n"
         "display(Markdown(f\"**Computed on:** {p['computed_on']}  \\n**Repo:** {p['repo']}  \\n\"\n"
         "                 f\"**Env:** {p['env']}  \\n**Protocol:** {p['protocol_pathA']}\"))\n"
         "tbl = '| setting | metric | value | note |\\n|---|---|---|---|\\n'\n"
         "for k, v in d['results'].items():\n"
         "    tbl += f\"| {k} | {v['metric']} | {v['value']} | {v['note'][:80]} |\\n\"\n"
         "display(Markdown(tbl))"),
    md("## Conclusion\n"
       "The released single checkpoint **leaks across folds** (F1 0.92); a **clean per-fold retrain "
       "reproduces ~0.57**, below the paper's 0.65. Our eCLIP routed through CORAL is ~chance (different "
       "task). Net: CORAL is a rigorous cold-start **existence proof** that interaction/family diversity "
       "enables generalization -- the motivation for our independent-family scaling (S1). It is NOT a "
       "drop-in baseline for eCLIP window-occupancy."),
]

# ── S3: RNA-memorization confound (is famfull family-transfer real?) ──────────────────────────────
S3 = [
    md("# S3 - Is the famfull family-transfer real, or RNA-memorization? (confound gate)\n"
       "**Question.** famfull shows held-out-FAMILY MCC 0.29-0.37 (S1). But are the RNAs held out? If a held "
       "(RNA, held-protein) pair can be scored from the RNA alone, the 'family transfer' is RNA-bindability "
       "memorization, not protein-family generalization.\n\n"
       "*Data: `mmpartnet_out/famfull_confounds.json` (CPU, reusable) + `mmpartnet_out/famfull/`.*"),
    md("## Definitions / math\n"
       "- **RNA overlap:** fraction of held RNAs that also appear in train (should be LOW for a clean claim).\n"
       "- **positive-RNA overlap:** fraction of held POSITIVE RNAs already positive (bound by some OTHER protein) in train.\n"
       "- **RNA-only bindability baseline** (protein IGNORED): "
       "$\\hat p(\\text{bound}\\mid r)=\\dfrac{\\#\\{(r,\\cdot)\\in\\text{train}: y=1\\}}{\\#\\{(r,\\cdot)\\in\\text{train}\\}}$; "
       "score a held pair $(r,p)$ by $\\hat p(\\text{bound}\\mid r)$ (threshold-swept).\n"
       "- **MCC** $=\\dfrac{TP\\,TN-FP\\,FN}{\\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$.\n"
       "- **Genuine protein signal (residual)** $=$ MCC(CORAL) $-$ MCC(RNA-only)."),
    code("import json; from pathlib import Path; from IPython.display import Markdown, display\n"
         "OUT = Path('..')/'..'/'mmpartnet_out'\n"
         "d = json.loads((OUT/'famfull_confounds.json').read_text())\n"
         "tab='| fold | RNA overlap | pos-RNA overlap | prot overlap | RNA-only MCC | CORAL MCC | residual |\\n'\n"
         "tab+='|---|---|---|---|---|---|---|\\n'\n"
         "for f in d['per_fold']:\n"
         "    res=f['coral_famfull_bestMCC']-f['rna_only_baseline_bestMCC']\n"
         "    tab+=(f\"| {f['fold']} | {f['rna_overlap_held_in_train']:.2f} | {f['pos_rna_overlap']:.2f} | \"\n"
         "          f\"{f['protein_overlap']:.3f} | {f['rna_only_baseline_bestMCC']:.3f} | {f['coral_famfull_bestMCC']:.3f} | {res:+.3f} |\\n\")\n"
         "display(Markdown(tab))\n"
         "s=d['summary']\n"
         "display(Markdown(f\"**Mean: RNA-only MCC {s['rna_only_baseline_bestMCC_mean']:.3f} vs CORAL {s['coral_famfull_bestMCC_mean']:.3f} \"\n"
         "                 f\"=> genuine protein residual ~{s['coral_famfull_bestMCC_mean']-s['rna_only_baseline_bestMCC_mean']:.3f} MCC. \"\n"
         "                 f\"RNA overlap {s['rna_overlap_mean']:.2f}.**\"))"),
    code("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np\n"
         "fo=[f['fold'] for f in d['per_fold']]; rna=[f['rna_only_baseline_bestMCC'] for f in d['per_fold']]\n"
         "cor=[f['coral_famfull_bestMCC'] for f in d['per_fold']]\n"
         "x=np.arange(len(fo)); w=0.38; fig,ax=plt.subplots(figsize=(7,4))\n"
         "ax.bar(x-w/2, rna, w, label='RNA-only baseline (protein ignored)', color='#c0504d')\n"
         "ax.bar(x+w/2, cor, w, label='CORAL famfull (RNA+protein)', color='#4472c4')\n"
         "ax.axhline(0.10, ls='--', c='gray', lw=1, label='famscale N<=200 (near-chance)')\n"
         "ax.set_xticks(x); ax.set_xticklabels([f'fold {k}' for k in fo]); ax.set_ylabel('best held-out-family MCC')\n"
         "ax.set_title('Family transfer vs RNA-memorization confound'); ax.legend(fontsize=8); fig.tight_layout()\n"
         "fig.savefig('S3_confound.png', dpi=110); plt.close(fig)\n"
         "from IPython.display import Image; display(Image('S3_confound.png'))"),
    md("## Conclusion\n"
       "The held **RNAs are ~100% shared** train<->held (only proteins are held out), and an **RNA-only "
       "bindability baseline reaches MCC ~0.23** (protein never seen) vs CORAL's 0.37. So **most of the famfull "
       "signal, and most of the famfull>famscale gain, is RNA-coverage memorization**, not protein-family "
       "transfer: more training families -> more training RNAs -> held RNAs better covered. The **genuine "
       "protein-family signal is the residual ~0.13 MCC** (best-epoch), highly fold-variable (~0 in folds 0/4, "
       "~0.2 in folds 1/3).\n\n"
       "**This does NOT confirm the interpolation hypothesis.** Two clean tests are needed (see plan + "
       "`S1_depeek_headline`, `S2_null_battery`): (1) **de-peek** the headline (report TEST at the val-selected "
       "epoch, not best-on-test); (2) a **protein-shuffle** control + an **RNA-AND-family-disjoint** split, to "
       "isolate the protein-transfer component. Until then, 'diversity is the lever' is **confounded by RNA "
       "coverage**.\n\n"
       "*Provenance: `mmpartnet_out/famfull_confounds.json` (repo `scripts/famfull_confounds.py`, CPU, CORAL fork data).*"),
]

# ── S1-depeek: the honest, no-test-peek family-transfer number ────────────────────────────────────
S1DP = [
    md("# S1-depeek - Honest de-peeked family-transfer number\n"
       "**Question.** The famfull headline (MCC 0.37) was best-epoch-ON-TEST (peeking) at max diversity. Under "
       "a proper no-test-peek protocol, what is the honest held-out-family number?\n\n"
       "*Protocol: 3-way FAMILY split per fold -- TRAIN(286) / EARLY-STOP-VAL(60, family-disjoint) / TEST(60, "
       "never used for selection); report TEST at the val-selected epoch. Data: `mmpartnet_out/famfull3_depeek.json`.*"),
    md("## Definitions / math\n"
       "- **test@val-epoch (HONEST):** TEST MCC at the epoch maximizing VAL-fold F1 (no test peeking).\n"
       "- **best-on-test (PEEK):** max TEST MCC over epochs (biased).\n"
       "- **peeking bias** = best-on-test - test@val-epoch.\n"
       "- **RNA-only floor** (S3): MCC an RNA-bindability baseline reaches ignoring the protein (= 0.234).\n"
       "- **protein-family signal** = honest test@val MCC - RNA floor."),
    code("import json; from pathlib import Path; from IPython.display import Markdown, display\n"
         "d = json.loads((Path('..')/'..'/'mmpartnet_out'/'famfull3_depeek.json').read_text())\n"
         "t = '| fold | best-val epoch | test@val MCC | Acc | last | best-peek |\\n|---|---|---|---|---|---|\\n'\n"
         "for f in d['per_fold']:\n"
         "    t += (f\"| {f['fold']} | {f['best_val_epoch']} | {f['test_MCC_at_val']:.3f} | \"\n"
         "          f\"{f['test_Acc_at_val']:.3f} | {f['test_MCC_last']:.3f} | {f['test_MCC_bestpeek']:.3f} |\\n\")\n"
         "display(Markdown(t)); s = d['summary']\n"
         "display(Markdown(f\"**Honest 5-fold test@val = {s['honest_test_MCC_mean']:.3f} +/- \"\n"
         "  f\"{s['honest_test_MCC_std']:.3f}** (Acc {s['honest_test_Acc_mean']:.2f}); best-on-test peek \"\n"
         "  f\"{s['bestpeek_MCC_mean']:.3f} (bias {s['peeking_bias']:.3f}); RNA-only floor {s['rna_only_floor']}; \"\n"
         "  f\"**protein-family signal above floor {s['protein_signal_above_rna_floor']:.3f} MCC**.\"))"),
    code("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np\n"
         "fo=[f['fold'] for f in d['per_fold']]; hv=[f['test_MCC_at_val'] for f in d['per_fold']]; pk=[f['test_MCC_bestpeek'] for f in d['per_fold']]\n"
         "x=np.arange(len(fo)); w=0.38; fig,ax=plt.subplots(figsize=(7,4))\n"
         "ax.bar(x-w/2, hv, w, label='honest (test@val-epoch)', color='#4472c4')\n"
         "ax.bar(x+w/2, pk, w, label='best-on-test (peek)', color='#c0504d')\n"
         "ax.axhline(0.234, ls='--', c='gray', lw=1, label='RNA-only floor (S3)')\n"
         "ax.set_xticks(x); ax.set_xticklabels([f'fold {k}' for k in fo]); ax.set_ylabel('held-out-family MCC')\n"
         "ax.set_title('De-peeked family transfer vs peek vs RNA floor'); ax.legend(fontsize=8); fig.tight_layout()\n"
         "fig.savefig('S1_depeek.png', dpi=110); plt.close(fig)\n"
         "from IPython.display import Image; display(Image('S1_depeek.png'))"),
    md("## Conclusion\n"
       "**Honest de-peeked held-out-family transfer = MCC 0.317 +/- 0.137** (Acc ~0.65). De-peeking does NOT "
       "collapse the number (peeking bias only ~0.06; honest 0.317 vs peek 0.378) -- so the famfull signal was "
       "not mainly a peeking artifact. BUT the RNA-only baseline already reaches 0.234 (S3), so the **genuine "
       "protein-family-transfer signal above the RNA-memorization floor is only ~0.08 MCC**, highly fold-variable "
       "(0.12-0.53). The best-val epoch is 0 for 3/5 folds -- training on families barely helps held-family "
       "transfer. Net: a real but small protein-family signal on a large RNA-coverage floor; the eye-catching "
       "0.37 was mostly RNA memorization, not clean interpolation. Mechanism follow-up: F1 near-vs-far.\n\n"
       "*Provenance: 3-way family split, CORAL model on a CUDA GPU (folds 0,2 salvaged from surviving adapters). "
       "`mmpartnet_out/famfull3_depeek.json` + per-fold `famfull3/*/test_metrics.csv`.*"),
]

# ── F1: near-vs-far mechanism (interpolation vs distance-independent) ──────────────────────────────
F1 = [
    md("# F1 - Near vs far: is the family-transfer near-neighbour interpolation?\n"
       "**Question.** The de-peeked protein-family residual (~0.08 MCC over the RNA floor) -- does it ride "
       "dense-neighbour *interpolation* (held families that have a close training relative)? If so, per-family "
       "MCC should rise with the nearest-train-family sequence identity.\n\n"
       "*Data: `mmpartnet_out/famfull3_nearvsfar.json` -- per held family (300 across 5 de-peek folds): MCC from "
       "the per-pair predictions vs nearest-train-family % identity (mmseqs easy-search).*"),
    md("## Definitions / math\n"
       "- **nearest_id(family)** = max % identity of a held-family representative to ANY training protein (mmseqs).\n"
       "- **per-family MCC** from that family's per-pair predictions (min 5 pairs).\n"
       "- **interpolation prediction:** MCC increases with nearest_id (positive Pearson/slope); far (isolated) "
       "families collapse toward chance.\n"
       "- **null (distance-independent):** flat MCC vs nearest_id; far approx near."),
    code("import json; from pathlib import Path; from IPython.display import Markdown, display\n"
         "d = json.loads((Path('..')/'..'/'mmpartnet_out'/'famfull3_nearvsfar.json').read_text()); s = d['summary']\n"
         "display(Markdown(f\"**n_families={s['n_families']}; Pearson(MCC, nearest_id) = {s['pearson_mcc_vs_nearest_id']:.3f} \"\n"
         "  f\"(slope {s['slope_mcc_per_pctid']:.4f}/%id). FAR (<30% id) n={s['far_lt30id_n']} meanMCC={s['far_lt30id_meanMCC']:.3f} \"\n"
         "  f\"vs NEAR (>=30%) n={s['near_ge30id_n']} meanMCC={s['near_ge30id_meanMCC']:.3f}.**\"))"),
    code("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np\n"
         "d = json.loads((Path('..')/'..'/'mmpartnet_out'/'famfull3_nearvsfar.json').read_text())\n"
         "x=np.array([r['nearest_id'] for r in d['per_family']]); y=np.array([r['mcc'] for r in d['per_family']])\n"
         "fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(x,y,s=14,alpha=0.5,color='#4472c4')\n"
         "if len(x)>2:\n"
         "    b,a=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),50); ax.plot(xs,b*xs+a,'r-',lw=1.5,label=f'slope={b:.4f}')\n"
         "ax.axvline(30,ls='--',c='gray',lw=1,label='30% id (far|near)'); ax.axhline(0,c='k',lw=0.5)\n"
         "ax.set_xlabel('nearest training-family % identity'); ax.set_ylabel('per-family MCC'); ax.legend(fontsize=8)\n"
         "ax.set_title('Held-family transfer vs distance to nearest training family'); fig.tight_layout()\n"
         "fig.savefig('F1_nearvsfar.png',dpi=110); plt.close(fig)\n"
         "from IPython.display import Image; display(Image('F1_nearvsfar.png'))"),
    md("## Conclusion\n"
       "The per-family transfer is **distance-INDEPENDENT**: Pearson(MCC, nearest_id) = -0.06 (flat/slightly "
       "negative slope), and the **far, isolated families (<30% identity to training, 283/300) transfer as well "
       "as the near ones** (far meanMCC 0.104 vs near 0.080). So the small protein-family signal is **NOT "
       "dense-neighbour interpolation** -- it does not require a close training relative. Because the held "
       "families are 30%-ID mmseqs clusters, almost all are already isolated, and they behave the same as the "
       "few near ones.\n\n"
       "Read together with S1-depeek + S3: the honest held-out-family number (0.317) is mostly RNA-coverage "
       "memorization (RNA floor 0.234); the residual protein-family signal (~0.08 MCC) is small, fold-variable, "
       "and **uniform across sequence distance** -- a weak, genuine, non-interpolative protein effect, not the "
       "diversity-driven breakthrough the raw curve suggested.\n\n"
       "*Caveat: per-family MCC is noisy at 5-9 pairs/family; the flat trend is the robust read, not any single "
       "family. Provenance: `mmpartnet_out/famfull3_nearvsfar.json` (repo `scripts/nearvsfar.py`, mmseqs).*"),
]

# ── F2: protein-derangement null (does the model USE the protein at all?) ──────────────────────────
F2 = [
    md("# F2 - Protein-derangement null: does the model USE the protein?\n"
       "**Question.** Given the honest de-peek (S1_depeek) is not distinguishable from an RNA-only baseline "
       "(S3), is the model even *conditioning* on the protein, or is it pure RNA memorization? Counterfactual: "
       "feed the WRONG protein for each held pair and see if the prediction survives.\n\n"
       "*Data: `mmpartnet_out/famfull3_derangement_null.json` (folds 0/2/4 -- the surviving saved adapters; "
       "1,3 tensors were deleted under the disk quota). Inference-only on a CUDA GPU.*"),
    md("## Definitions / math\n"
       "- **Derangement** $\\sigma$: a fixed-point-free permutation of the protein assignment across held pairs "
       "(no pair keeps its true protein), RNA window + ORIGINAL label unchanged. 3 seeds/fold.\n"
       "- **deranged MCC** = MCC(model prediction on $(RNA, \\sigma(\\text{protein}))$, ORIGINAL label).\n"
       "- **drop** = honest - deranged.\n"
       "- **Read:** drop $\\approx 0$ (deranged $\\approx$ honest) -> protein IGNORED (pure RNA memorization); "
       "drop large / deranged -> chance -> the prediction DEPENDS on the protein input. NB this tests protein "
       "USE, not protein-family TRANSFER (a model that memorized seen (protein,RNA) pairs also collapses here)."),
    code("import json; from pathlib import Path; from IPython.display import Markdown, display\n"
         "d = json.loads((Path('..')/'..'/'mmpartnet_out'/'famfull3_derangement_null.json').read_text())\n"
         "t = '| fold | honest MCC | deranged-protein MCC (3 seeds) | drop |\\n|---|---|---|---|\\n'\n"
         "hs=[]; ds=[]\n"
         "for k in sorted(d, key=int):\n"
         "    v=d[k]; hs.append(v['honest']); ds.append(v['deranged_mean'])\n"
         "    t += f\"| {k} | {v['honest']:.3f} | {v['deranged_mean']:.3f} ({', '.join(f'{x:.3f}' for x in v['deranged'])}) | {v['honest']-v['deranged_mean']:+.3f} |\\n\"\n"
         "display(Markdown(t))\n"
         "import statistics as st\n"
         "display(Markdown(f\"**Mean honest {st.mean(hs):.3f} vs deranged {st.mean(ds):.3f} (drop {st.mean(hs)-st.mean(ds):+.3f}); RNA-only floor 0.234.** \"\n"
         "  f\"Deranged collapses to ~0 -> the model DOES condition on the protein (not pure RNA memorization).\"))"),
    code("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np\n"
         "ks=sorted(d,key=int); hs=[d[k]['honest'] for k in ks]; ds=[d[k]['deranged_mean'] for k in ks]\n"
         "x=np.arange(len(ks)); w=0.38; fig,ax=plt.subplots(figsize=(7,4))\n"
         "ax.bar(x-w/2,hs,w,label='honest (true protein)',color='#4472c4'); ax.bar(x+w/2,ds,w,label='deranged protein',color='#c0504d')\n"
         "ax.axhline(0.234,ls='--',c='gray',lw=1,label='RNA-only floor'); ax.axhline(0,c='k',lw=0.5)\n"
         "ax.set_xticks(x); ax.set_xticklabels([f'fold {k}' for k in ks]); ax.set_ylabel('held-out-family MCC')\n"
         "ax.set_title('Protein-derangement null: prediction collapses without the true protein'); ax.legend(fontsize=8); fig.tight_layout()\n"
         "fig.savefig('F2_derangement.png',dpi=110); plt.close(fig)\n"
         "from IPython.display import Image; display(Image('F2_derangement.png'))"),
    md("## Conclusion (corrects the earlier read)\n"
       "Feeding the WRONG protein **collapses MCC from ~0.25 to ~0.03** (below the RNA floor). So the model **is "
       "conditioning on the protein** -- it is NOT pure RNA memorization, and the earlier 'protein ignored' "
       "framing was too strong. **But this refutes 'ignored', not 'no transfer'**: the derangement tests protein "
       "USE, and a model that memorized seen (protein,RNA) associations collapses here too. Read together with "
       "S1_depeek (protein-using model does NOT beat the protein-blind RNA-only baseline, 0.317 vs 0.234) and F1 "
       "(effect is flat vs family distance), the precise conclusion is: **the model uses protein IDENTITY as a "
       "consistency/lookup key on seen structure, not protein-FAMILY structure for generalization** -- used-but-"
       "non-generalizing. Caveat: folds 0/2/4 only (1/3 adapters lost to quota); tests use, not transfer.\n\n"
       "*Provenance: deranged-protein inference on saved best_adapters, CUDA GPU; `mmpartnet_out/famfull3_derangement_null.json`.*"),
]

# ── PB1: ProofBind verification of the whole conclusion (binding + significance) ──────────────────
PB1 = [
    md("# PB1 - ProofBind verification: is the family-transfer conclusion bound AND significant?\n"
       "**Question.** Apply the ProofBind discipline to our own headline: (1) is MCC 0.317 correctly computed "
       "(binding), and (2) is the protein-family-transfer CLAIM statistically established? Re-derives everything "
       "from the COMMITTED raw artifacts, independently of the scripts that produced them.\n\n"
       "*Data: per-fold `mmpartnet_out/famfull3/*/{val,test}_metrics.csv` + `test_preds.csv`; "
       "`famfull3_depeek.json`, `famfull_confounds.json`, `famfull3_nearvsfar.json`.*"),
    md("## Definitions / math\n"
       "- **MCC** $=\\frac{TP\\,TN-FP\\,FN}{\\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$.\n"
       "- **Binding (B1-B4):** B1 the reported best epoch = independent $\\arg\\max_e$ VAL-F1; B2 test-metrics MCC "
       "at that epoch = reported; B3 a SECOND path (predict.py per-pair preds) agrees; B4 label-shuffle "
       "counterexample MUST fire (MCC$\\to 0$).\n"
       "- **Honest de-peek** = mean over folds of test MCC @ val-selected epoch; fold-bootstrap 95% CI.\n"
       "- **Paired residual** = per-fold (honest - fold-matched RNA-only); sign test + bootstrap CI.\n"
       "- **Near-vs-far** = family-block permutation p for Pearson(per-family MCC, nearest-train %id)."),
    code("import json, csv, math, random\n"
         "from pathlib import Path\n"
         "import numpy as np\n"
         "from IPython.display import Markdown, display\n"
         "OUT = Path('..')/'..'/'mmpartnet_out'; FF = OUT/'famfull3'\n"
         "def mcc(pred,y):\n"
         "    pred=np.asarray(pred); y=np.asarray(y)\n"
         "    tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())\n"
         "    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (tp*tn-fp*fn)/d if d else 0.0\n"
         "dep={f['fold']:f for f in json.loads((OUT/'famfull3_depeek.json').read_text())['per_fold']}\n"
         "tab='| fold | argmax VAL-ep | reported best-ep | MCC@val (metrics) | MCC (preds path) | label-shuffle |\\n|---|---|---|---|---|---|\\n'\n"
         "allok=True\n"
         "for K in range(5):\n"
         "    vm=list(csv.DictReader(open(FF/str(K)/'val_metrics.csv'))); tm=list(csv.DictReader(open(FF/str(K)/'test_metrics.csv'))); pr=list(csv.DictReader(open(FF/str(K)/'test_preds.csv')))\n"
         "    amax=max(((int(r['Epoch']),float(r['F1'])) for r in vm),key=lambda x:x[1])[0]\n"
         "    mccv={int(r['Epoch']):float(r['MCC']) for r in tm}[amax]\n"
         "    y=[int(float(r['true_label'])) for r in pr]; pp=[int(float(r['prediction'])) for r in pr]\n"
         "    mp=mcc(pp,y); ys=y[:]; random.Random(0).shuffle(ys); msh=mcc(pp,ys)\n"
         "    b=(amax==dep[K]['best_val_epoch']) and abs(mccv-dep[K]['test_MCC_at_val'])<1e-6 and abs(mp-dep[K]['test_MCC_at_val'])<0.02 and abs(msh)<0.05\n"
         "    allok=allok and b\n"
         "    tab+=f\"| {K} | {amax} | {dep[K]['best_val_epoch']} | {mccv:.3f} | {mp:.3f} | {msh:+.3f} |\\n\"\n"
         "display(Markdown(tab)); display(Markdown(f'**Binding B1-B4: {\"ALL PASS\" if allok else \"FAIL\"}** (val-selection genuine; two code paths agree; label-shuffle collapses -> metric real).'))"),
    code("# significance: de-peek CI + paired residual vs fold-matched RNA-only\n"
         "conf={f['fold']:f['rna_only_baseline_bestMCC'] for f in json.loads((OUT/'famfull_confounds.json').read_text())['per_fold']}\n"
         "hon=np.array([dep[k]['test_MCC_at_val'] for k in range(5)]); rna=np.array([conf[k] for k in range(5)]); diff=hon-rna\n"
         "rng=np.random.default_rng(0)\n"
         "bh=[np.mean(rng.choice(hon,5,True)) for _ in range(10000)]; bd=[np.mean(rng.choice(diff,5,True)) for _ in range(10000)]\n"
         "hlo,hhi=np.percentile(bh,[2.5,97.5]); dlo,dhi=np.percentile(bd,[2.5,97.5]); npos=int((diff>0).sum())\n"
         "display(Markdown(f'**Honest de-peek MCC = {hon.mean():.3f}, 95% CI [{hlo:.3f},{hhi:.3f}].** '\n"
         "  f'Paired residual over fold-matched RNA-only = {diff.mean():+.3f}, 95% CI [{dlo:+.3f},{dhi:+.3f}] '\n"
         "  f'({npos}/5 folds positive) -> **{\"EXCLUDES 0\" if dlo>0 else \"INCLUDES 0 (NOT significant)\"}**.'))\n"
         "nf=json.loads((OUT/'famfull3_nearvsfar.json').read_text())['per_family']\n"
         "x=np.array([r['nearest_id'] for r in nf]); yv=np.array([r['mcc'] for r in nf]); obs=np.corrcoef(x,yv)[0,1]\n"
         "null=np.array([np.corrcoef(x,rng.permutation(yv))[0,1] for _ in range(10000)]); pp=(np.sum(np.abs(null)>=abs(obs))+1)/(len(null)+1)\n"
         "display(Markdown(f'**Near-vs-far:** Pearson(MCC, nearest-train %id) = {obs:.3f}, family-block permutation p = {pp:.3f} -> {\"distance-dependent\" if pp<0.05 else \"NO distance dependence (interpolation refuted)\"}.'))"),
    md("## Conclusion -- ProofBind verdict\n"
       "**MECHANICALLY BOUND, INTERPRETIVELY NOT-ESTABLISHED.** The number is correct (B1-B4 pass; label-shuffle "
       "fires). But the protein-family-transfer CLAIM fails: the honest de-peek (0.317) is not separable from a "
       "protein-blind RNA-only baseline (paired residual +0.08, CI includes 0, 2/5 folds), and the effect is "
       "distance-independent (permutation p~0.29). The protein-derangement null (F2) shows the model *does* "
       "condition on the protein, so the precise statement is **used-but-non-generalizing**: protein identity as "
       "a lookup key, not protein-family structure.\n\n"
       "**Tightest honest claim:** on a protein-family-disjoint, de-peeked split the CORAL interaction model "
       "scores MCC 0.317 [0.196,0.443], not distinguishable from RNA-bindability memorization; no protein-family-"
       "transfer effect is established (n=5 -> absence of evidence). Decisive open tests: RNA+family doubly-"
       "disjoint split; M2 profile-shape on a leave-out-pretrained PARNET body."),
]

if __name__ == "__main__":
    jobs = [("leakage", "L1_decomposition", L1),
            ("scaling", "S1_family_scaling", S1),
            ("scaling", "S1_depeek_headline", S1DP),
            ("scaling", "S3_rna_overlap", S3),
            ("scaling", "F1_near_vs_far", F1),
            ("scaling", "F2_protein_derangement", F2),
            ("scaling", "PB1_proofbind_verification", PB1),
            ("coral", "C1_coral_reproduction", C1)]
    for sub, name, cells in jobs:
        build(ROOT / "notebooks" / sub / f"{name}.ipynb",
              ROOT / "notebooks" / sub / "executed" / f"{name}_executed.ipynb",
              cells, timeout=300)
