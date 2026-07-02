"""Notebook 18 - k-fold-over-RBPs head bake-off: can a gauge-fixed BioPWM x per-residue combination MATCH
per-residue M2 profile performance while staying interpretable? per-residue (reference) vs BioPWM (null) vs
Form D (RNA envelope + protein PWM) vs Form D' (+ info-bottleneck protein->broad channel). Every RBP held out
once; per-RBP profile-Pearson gap vs shuffle + within-family, paired Wilcoxon + bootstrap. Paper + reg-genomics
style. Executed; targets the merge worktree."""
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
    "sys.path.insert(0,str(REPO/'scripts')); import plot_style as ps; ps.apply_style()\n"
    "OUT=REPO/'mmpartnet_out'; FIGD=REPO/'notebooks'/'demo'/'executed'\n"
    "def J(n):\n"
    "    p=OUT/n; return json.loads(p.read_text()) if p.exists() else None\n"
    "def show(fig,name,sup=None):\n"
    "    if sup: fig.suptitle(sup,fontsize=11,fontweight='bold',y=1.02)\n"
    "    fig.savefig(str(FIGD/name),bbox_inches='tight',dpi=200); plt.close(fig); display(_Img(filename=str(FIGD/name)))\n"
    "LAB={'perres':'per-residue\\nX-attn (ref)','biopwm':'BioPWM\\n(null)','envpwm':\"Form D\\nenv+PWM\",'envpwm_z':\"Form D'\\nenv+PWM+IB\"}\n"
)
DATA = ("**Data.** M2 leave-out-RBP zero-shot, real lab PARNET + eCLIP. **5-fold over RBPs** (every RBP held out "
        "exactly once = full zero-shot coverage, fixing the prior single-30%-split n). Per-RBP profile-Pearson; "
        "gap = protein minus shuffled-protein and minus within-family shuffle; paired Wilcoxon + bootstrap CI over "
        "the pooled RBP set. Forms D/D' are gauge-fixed: a protein-AGNOSTIC RNA envelope + a Stormo PWM (D), plus "
        "(D') an information-bottlenecked (<=R bits) protein->broad channel, so the PWM is the only channel that "
        "can carry sharp protein-conditional structure.")
ATTR = "\n\nClaude-assisted. Forms D/D' per the combine-design workflow (softmax-gauge identifiability)."

build(DEMO / "18_kfold_bakeoff.ipynb", EX / "18_kfold_bakeoff_executed.ipynb", [
    md("# 18 - k-fold head bake-off: matching per-residue performance with an interpretable head\n\n"
       "**What.** Proper 5-fold-over-RBPs comparison of the M2 zero-shot heads: can a **gauge-fixed BioPWM x "
       "per-residue** combination (Forms D/D') match the per-residue cross-attn *performance* while keeping a "
       "**readable PWM** (which per-residue attention does not provide - it is anti-faithful zero-shot)?\n\n"
       "**Why.** per-residue wins the profile but its attention does not localize zero-shot (notebook 15); BioPWM "
       "is interpretable but null on the profile (notebook 16). Forms D/D' split the prediction into a "
       "protein-agnostic RNA envelope (the coarse coverage BioPWM lacked) + a protein-specific Stormo PWM (sharp, "
       "interpretable), with the softmax gauge closed so the protein signal is attributable to the motif by "
       "algebra.\n\n" + DATA),
    md("## Definitions\n\nTarget logit (Form D'): $t(x)=\\mathrm{softplus}(\\beta)\\,m_{\\mathrm{PWM}}(x)+g_{\\mathrm{env}}(H)(x)+z(P)^\\top\\phi(x)$. "
       "$m_{\\mathrm{PWM}}$ = Stormo log-odds occupancy of the protein-generated PWM on the raw RNA (sharp, "
       "protein-specific, **interpretable**); $g_{\\mathrm{env}}(H)$ = protein-AGNOSTIC RNA coverage envelope from "
       "PARNET features; $z(P)^\\top\\phi$ = band-limited ($\\phi$ = low-freq cosine basis), KL-bit-capped "
       "protein->broad channel. Form D drops $z$. Because $g_{\\mathrm{env}}$ has no protein input and $z$ is "
       "bit-capped, the leakage-free gap $t_{\\mathrm{real}}-t_{\\mathrm{shuf}}=\\mathrm{softplus}(\\beta)\\,\\Delta m_{\\mathrm{PWM}}$ "
       "(+ a certified $\\le R$-bit broad term) - attributable to the readable motif."),

    code(HEAD + "import subprocess\n"
         "# pool the folds if the pooled file is absent (idempotent)\n"
         "if J('m2_kfold_HepG2.json') is None:\n"
         "    import mmpartnet.experiments.m2_kfold_pool as P; sys.argv=['x','HepG2','5']\n"
         "    try: P.main()\n"
         "    except Exception as e: print('pool note', e)\n"
         "d=J('m2_kfold_HepG2.json'); A=d['archs']\n"
         "for a in ['perres','biopwm','envpwm','envpwm_z']:\n"
         "    if a in A: print(f\"  {a:10} real {A[a]['real']:.3f} | gap vs shuffle {A[a]['gap_der']:+.4f} | vs within-family {A[a]['gap_fam']:+.4f} ({A[a]['n_beat_fam']}/{A[a]['n']}, p={A[a]['wilcoxon_fam']:.1e})\" + (f\" | real vs perres {A[a].get('vs_perres_real',0):+.4f}\" if a!='perres' else ''))"),
    code(HEAD + "d=J('m2_kfold_HepG2.json'); A=d['archs']; order=[a for a in ['perres','biopwm','envpwm','envpwm_z'] if a in A]\n"
         "fig,(a1,a2)=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw={'width_ratios':[1.1,1]})\n"
         "ps.gap_violin(a1,{LAB[a]:[r['pearson_real']-r['pearson_fam'] for r in A[a]['rows']] for a in order},ylabel='per-RBP gap vs within-family',title='Within-family gap (the hard control)',paired=False,ref=0)\n"
         "x=np.arange(len(order))\n"
         "a2.bar(x-0.2,[A[a]['gap_der'] for a in order],0.38,yerr=[[A[a]['gap_der']-A[a]['gap_der_ci'][0] for a in order],[A[a]['gap_der_ci'][1]-A[a]['gap_der'] for a in order]],capsize=3,color=ps.PALETTE['protein'],label='vs shuffle',edgecolor='white')\n"
         "a2.bar(x+0.2,[A[a]['gap_fam'] for a in order],0.38,yerr=[[A[a]['gap_fam']-A[a]['gap_fam_ci'][0] for a in order],[A[a]['gap_fam_ci'][1]-A[a]['gap_fam'] for a in order]],capsize=3,color=ps.PALETTE['family'],label='vs within-family',edgecolor='white')\n"
         "if 'perres' in A: a2.axhline(A['perres']['gap_fam'],color=ps.PALETTE['rna_only'],ls='--',lw=1,label='perres within-family')\n"
         "a2.axhline(0,color='#000',lw=0.8); a2.set_xticks(x); a2.set_xticklabels([LAB[a] for a in order],fontsize=7); a2.set_ylabel('M2 profile-Pearson gap'); a2.legend(frameon=True,fontsize=6.5); ps.despine(a2)\n"
         "ps.panel_label(a1,'a'); ps.panel_label(a2,'b')\n"
         "show(fig,'nb18_kfold.png','5-fold M2 zero-shot: does a gauge-fixed interpretable head match per-residue?')"),
    code(HEAD + "d=J('m2_kfold_HepG2.json'); A=d['archs']\n"
         "best=max([a for a in A if a!='perres'],key=lambda a:A[a]['gap_fam'])\n"
         "pr=A.get('perres',{})\n"
         "display(Markdown(f'''**Result (5-fold, HepG2).** per-residue (reference): real {pr.get('real',0):.3f}, "
         "within-family gap {pr.get('gap_fam',0):+.4f}. BioPWM: {A['biopwm']['gap_fam']:+.4f} (null on profile, as "
         "expected). **Form D (env+PWM): {A['envpwm']['gap_fam']:+.4f}**; **Form D' (env+PWM+IB): "
         "{A['envpwm_z']['gap_fam']:+.4f}** (real vs perres {A['envpwm_z'].get('vs_perres_real',0):+.4f}). "
         "Best interpretable head = **{best}**. The question - does an interpretable (readable-PWM) head reach "
         "per-residue within-family performance - is answered by whether {best}'s gap_fam CI "
         "[{A[best]['gap_fam_ci'][0]:+.4f},{A[best]['gap_fam_ci'][1]:+.4f}] overlaps per-residue "
         "{pr.get('gap_fam',0):+.4f}.'''))"),

    md("## Conclusion\n\n"
       "The bake-off tests whether the **interpretability tax is zero**: Forms D/D' keep a readable Stormo PWM (the "
       "protein signal is attributable to it by algebra, unlike per-residue attention which collapses zero-shot) and "
       "add a protein-agnostic RNA envelope to recover the coarse coverage BioPWM lacked. If Form D' (the "
       "information-bottlenecked variant) reaches per-residue within-family performance, we get **per-residue-level "
       "M2 zero-shot with BioPWM-grade interpretability** - the goal. The D'-vs-D delta isolates how much protein "
       "signal is broad-but-not-motif; the B3 PWM-ablation certificate (shuffle the PWM -> gap must collapse) gates "
       "the interpretability claim. K562 replication + the certificate follow." + ATTR),
])
print("NB18 KFOLD DONE")
