"""Notebook 17 - cross-method-family bake-off: best of each family (colleagues' + mine), scaled + ablated, on the
common leakage-controlled lab panel. HONEST about axes: PARNET-conditioned families are scored vs the (leaked)
RNA-only baseline; BioPWM is scored vs shuffled-protein (leakage-free) - different baselines, shown side by side
with the leakage status explicit. Paper-grade via plot_style + viz. Executed; targets the merge worktree."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nbgen import md, code, build

W = Path(r"D:/FOAM2.0/poc/ml4rg_parnet/dist/mmparnet-merge")
DEMO = W / "notebooks" / "demo"; EX = DEMO / "executed"
HEAD = (
    "import os, sys, json, pathlib\nimport numpy as np, matplotlib.pyplot as plt\n"
    "from IPython.display import Markdown, display, Image as _Img\n_here=pathlib.Path.cwd().resolve()\n"
    "REPO=next((c for c in (_here,*_here.parents) if (c/'src'/'mmpartnet').is_dir()),_here)\n"
    "sys.path.insert(0,str(REPO/'scripts')); import plot_style as ps, viz; ps.apply_style()\n"
    "OUT=REPO/'mmpartnet_out'; FIGD=REPO/'notebooks'/'demo'/'executed'\n"
    "def J(n):\n"
    "    p=OUT/n; return json.loads(p.read_text()) if p.exists() else None\n"
    "def Jf(*names):\n"
    "    for n in names:\n"
    "        d=J(n)\n"
    "        if d is not None: return d\n"
    "    return None\n"
    "def show(fig,name,sup=None):\n"
    "    if sup: fig.suptitle(sup,fontsize=11,fontweight='bold',y=1.02)\n"
    "    fig.savefig(str(FIGD/name),bbox_inches='tight',dpi=200); plt.close(fig); display(_Img(filename=str(FIGD/name)))\n"
)
DATA = ("**Data.** Common lab panel (K=68 eCLIP tracks / 45 RBPs), real PARNET `parnet.7m-0.0`, per-residue ESM, "
        "5 seeds, leakage-controlled `eval_controls` harness (RNA-only multitask baseline + random-body leakage "
        "control + cross-family/within-family protein-shuffle + paired sign tests).")
ATTR = "\n\nClaude-assisted. concat/early-fusion = Christoph; PARNET-eval baselines = gudiyi; per-residue cross-attn + BioPWM = ours."

build(DEMO / "17_method_bakeoff.ipynb", EX / "17_method_bakeoff_executed.ipynb", [
    md("# 17 - Cross-method-family bake-off (colleagues + ours), scaled & ablated\n\n"
       "**What.** One leakage-controlled harness, the best of each method family side by side: the RNA-only "
       "baseline + random-body leakage control; **early-fusion/concat** (Christoph); **FiLM**; **pooled "
       "cross-attention**; **per-residue cross-attention** (ours, the M1/M2 winner); and **BioPWM** (ours, "
       "leakage-free). Plus each family's ablation summary and the interpretability axis.\n\n"
       "**Why - read the axes carefully.** The PARNET-conditioned families are scored as a gap vs the RNA-only "
       "multitask baseline, which is ~47% frozen-PARNET leakage (random-body control) - a *leaked* yardstick. "
       "BioPWM is scored vs a shuffled protein with NO PARNET - a *leakage-free* yardstick. They are NOT the same "
       "number; the honest comparison is per-axis, with leakage status explicit.\n\n" + DATA),

    md("## 1 - PARNET-conditioned families vs the (leaked) RNA-only baseline"),
    code(HEAD + "f=Jf('binding_x_x.json','binding_fair.json'); b=f['baselines']\n"
         "leak=f.get('leakage_attributable_auprc')\n"
         "print(f\"RNA-only multitask {b['rna_only_multitask']:.4f} | random-body {b.get('rna_only_randombody')} \"\n"
         "      f\"=> leakage {leak:+.4f} (~{100*leak/b['rna_only_multitask']:.0f}% of the baseline is PARNET memorization)\")\n"
         "for m,v in f['methods'].items(): print(f\"  {m:8} vs RNA-only {v['gap_vs_rna_only']:+.4f} | {v['direction_vs_rna_only']} {v['n_beat_rna_only']}/{v['n_rbp']} (p={v['sign_test_binom_p']:.1e})\")"),
    code(HEAD + "show(viz.fig_fair(Jf('binding_x_x.json','binding_fair.json')),'nb17_families.png','Conditioning families vs the RNA-only baseline (per-residue is the only winner; baseline is ~47% leaked)')"),

    md("## 2 - BioPWM on its own (leakage-free) axis"),
    code(HEAD + "di=Jf('binding_biopwm_recog_indist.json','binding_biopwm_indist.json'); dr=Jf('binding_biopwm_recog_rbp.json','binding_biopwm_rbp.json')\n"
         "fig,ax=plt.subplots(figsize=(5.2,3.6)); rows=[]\n"
         "for t,d in [('in-dist\\nvs cross-fam',di,'gap'),('in-dist\\nvs within-fam',di,'gap_fam'),('zero-shot\\nvs cross-fam',dr,'gap'),('zero-shot\\nvs within-fam',dr,'gap_fam')] if False else []:\n"
         "    pass\n"
         "specs=[('in-dist\\ncross-fam',di,'gap'),('in-dist\\nwithin-fam',di,'gap_fam'),('zero-shot\\ncross-fam',dr,'gap'),('zero-shot\\nwithin-fam',dr,'gap_fam')]\n"
         "x=np.arange(len(specs)); vals=[s[1][s[2]] if s[1] else 0 for s in specs]\n"
         "cols=[ps.PALETTE['protein'] if 'cross' in s[0] else ps.PALETTE['family'] for s in specs]\n"
         "ax.bar(x,vals,0.6,color=cols,edgecolor='white')\n"
         "ax.axhline(0,color='#000',lw=0.8); ax.set_xticks(x); ax.set_xticklabels([s[0] for s in specs],fontsize=7)\n"
         "ax.set_ylabel('protein-vs-shuffle auPRC gap (LEAKAGE-FREE)'); ps.despine(ax)\n"
         "show(fig,'nb17_biopwm.png','BioPWM: leakage-free protein-vs-shuffle gap (no PARNET in the loop)')"),

    md("## 3 - The bake-off table (best of each family, with axis + leakage + interpretability)"),
    code(HEAD + "f=Jf('binding_x_x.json','binding_fair.json'); M=f['methods']\n"
         "di=Jf('binding_biopwm_recog_indist.json','binding_biopwm_indist.json'); dr=Jf('binding_biopwm_recog_rbp.json','binding_biopwm_rbp.json')\n"
         "def g(m): return M[m]['gap_vs_rna_only'] if m in M else float('nan')\n"
         "rowsmd=[\n"
         " '| family (owner) | best method | metric / baseline | gap | leakage | interpretability |',\n"
         " '|---|---|---|---|---|---|',\n"
         " f'| early-fusion (Christoph) | concat | auPRC vs RNA-only | {g(\"concat\"):+.4f} | leaked baseline | none |',\n"
         " f'| FiLM (ours) | film | auPRC vs RNA-only | {g(\"film\"):+.4f} | leaked baseline | low |',\n"
         " f'| pooled cross-attn (ours) | xattn | auPRC vs RNA-only | {g(\"xattn\"):+.4f} | leaked baseline | attention (in-dist) |',\n"
         " f'| **per-residue cross-attn (ours)** | perres | auPRC vs RNA-only | **{g(\"perres\"):+.4f}** | leaked baseline | attention (collapses zero-shot) |',\n"
         " f'| **BioPWM (ours)** | recog | auPRC vs **shuffle** | **{di[\"gap\"]:+.4f}** in-dist / **{dr[\"gap\"]:+.4f}** zero-shot | **LEAKAGE-FREE** | **explicit PWM** |',\n"
         "]\n"
         "display(Markdown(chr(10).join(rowsmd)))"),

    md("## Conclusion - who wins which axis\n\n"
       "- **On the leaked in-distribution auPRC** (the arena CORAL is chartered in): conditioning is a ladder "
       "concat<FiLM<pooled-X-attn<**per-residue X-attn** (+0.016), but the baseline is ~47% PARNET leakage, so a "
       "'win' here is leakage-confounded - and richness/capacity are a closed null lever (notebooks 14-15).\n"
       "- **On the leakage-free axis**: **BioPWM** is the only method that removes PARNET from the loop, with a "
       "large protein-vs-shuffle gap (in-dist + zero-shot) that even beats the within-family shuffle - RBP-specific, "
       "clean.\n"
       "- **On interpretability under shift**: only BioPWM's explicit PWM survives zero-shot (attention collapses, "
       "notebook 15); and the protein deviation is independently validated on fine affinity assays (RBNS/RNAcompete, "
       "notebook 16).\n\n"
       "**The defensible bake-off verdict:** per-residue cross-attention is the best *in-distribution conditioning* "
       "but on a leaked yardstick; **BioPWM is the best *leakage-controlled + interpretable* method**, which is the "
       "axis that actually differentiates from CORAL. Ablations across families converge on one rule (Ch207, now "
       "multi-arm validated): the lever is supervision + per-residue position-resolution + leakage-free readout, "
       "NOT model capacity/richness. The one remaining unblock is a leave-out-pretrained PARNET checkpoint (Moyon) "
       "to de-leak the in-distribution axis itself." + ATTR),
])
print("NB17 BAKEOFF DONE")
