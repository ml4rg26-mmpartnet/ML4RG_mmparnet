"""Build the recent-findings notebooks (executed) in the shared thin/rigorous style: ONE question ->
definitions -> load committed result JSON (computed on a GPU node) -> display -> conclusion + leakage
caveat. Results are loaded from mmpartnet_out/ (the committed shared space). Where a number needs the
CORAL repo / a GPU to recompute, the notebook DISPLAYS the recorded 5090-node result and states how it
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
    code("# family-scaling curve from the committed 5090 results (mmpartnet_out/famscale/famscale_N/0/)\n"
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

if __name__ == "__main__":
    jobs = [("leakage", "L1_decomposition", L1),
            ("scaling", "S1_family_scaling", S1),
            ("scaling", "S3_rna_overlap", S3),
            ("coral", "C1_coral_reproduction", C1)]
    for sub, name, cells in jobs:
        build(ROOT / "notebooks" / sub / f"{name}.ipynb",
              ROOT / "notebooks" / sub / "executed" / f"{name}_executed.ipynb",
              cells, timeout=300)
