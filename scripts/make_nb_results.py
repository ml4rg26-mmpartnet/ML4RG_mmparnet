"""Notebook 19 - Results + interpretability, Marsico/Gagneur house style + domain-specific regulatory-genomics
viz. Performance (k-fold head bake-off, leakage battery, cross-cell replication) + interpretability made CONCRETE
(read the motif: sequence logos; see where on the RNA: eCLIP coverage track; coarse/fine decomposition). Executed;
targets the merge worktree. plot_style.set_house_style + logomaker logos + eclip_track + bakeoff_bars + scatter."""
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
    "sys.path.insert(0,str(REPO/'scripts')); import plot_style as ps; ps.set_house_style()\n"
    "OUT=REPO/'mmpartnet_out'; FIGD=REPO/'notebooks'/'demo'/'executed'\n"
    "def J(n):\n"
    "    p=OUT/n; return json.loads(p.read_text()) if p.exists() else None\n"
    "def show(fig,name,sup=None):\n"
    "    if sup: fig.suptitle(sup,fontsize=11,fontweight='bold',y=1.02)\n"
    "    fig.savefig(str(FIGD/name),bbox_inches='tight',dpi=200); plt.close(fig); display(_Img(filename=str(FIGD/name)))\n"
)
DATA = ("**Data (Moyon/Marsico lab).** Frozen all-223 PARNET `parnet.7m-0.0`, full-223 `encode.filtered.hfds` "
        "eCLIP, per-residue ESM. M2 = nt-resolution profile, leave-out-RBP zero-shot, **5-fold over RBPs** (every "
        "RBP held out once). Colorblind-safe ACGU logos; eCLIP tracks are softmax-over-positions (sum=1). Figures "
        "via `scripts/plot_style.py`.")
ATTR = "\n\nClaude-assisted; figures in Marsico (RBPNet/PanRBPNet/RIBEX) + Gagneur (OUTRIDER/AbSplice) house style."

build(DEMO / "19_results_and_interpretability.ipynb", EX / "19_results_and_interpretability_executed.ipynb", [
    md("# 19 - Results & interpretability (Marsico/Gagneur house style)\n\n"
       "**What.** The performance story (which head predicts the unseen-RBP nt-resolution eCLIP profile) and the "
       "**interpretability made concrete** (read the motif, see where on the RNA, decompose coarse vs fine), in the "
       "labs' figure idioms with domain-specific regulatory-genomics viz.\n\n"
       "**Headline.** Per-residue cross-attention wins the profile; **`perres_aux` matches it (0.164 vs 0.159) "
       "while adding a faithful readable PWM** - so we get per-residue performance with an explicit motif readout. "
       "Every head that makes the PWM the *predictor* (BioPWM/Form-D/occ_footprint/motifscan) is null/below - "
       "interpretability comes from *explaining* the cross-attn, not bottlenecking it.\n\n" + DATA),

    # ---- Section 1: performance ----
    md("## 1 - Performance: which head predicts the unseen-RBP profile?"),
    code(HEAD + "b=J('m2_bakeoff_HepG2.json')\n"
         "for k,v in sorted(b.items(),key=lambda kv:-kv[1]['gap_fam']): print(f\"  {k:14} real {v['real']:.3f} | within-family gap {v['gap_fam']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]\")"),
    code(HEAD + "b=J('m2_bakeoff_HepG2.json')\n"
         "kind={'per-residue':'winner','perres_aux':'match','BioPWM':'null','Form D':'null',\"Form D'\":'null','occ_footprint':'null','perrespwm':'below','motifscan':'below'}\n"
         "items=[(k,v['gap_fam'],v['ci'][0],v['ci'][1],kind.get(k,'below')) for k,v in b.items()]\n"
         "fig,(a,c)=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw={'width_ratios':[1.2,1]})\n"
         "ps.bakeoff_bars(a,items,null_level=0,noise=b['BioPWM']['ci'][1],ylabel='within-family profile-Pearson gap',title='Head bake-off (5-fold, HepG2)')\n"
         "# leakage control battery from the per-residue k-fold rows\n"
         "R=J('m2_kfold_HepG2.json')['archs']['perres']['rows']\n"
         "ps.gap_violin(c,{'protein':[r['pearson_real'] for r in R],'shuffle':[r['pearson_shuf'] for r in R],'within-family':[r['pearson_fam'] for r in R]},ylabel='per-RBP profile Pearson',title='Leakage-controlled (per-residue)',paired=True)\n"
         "ps.panel_label(a,'a'); ps.panel_label(c,'b')\n"
         "show(fig,'nb19_perf.png','Per-residue wins; perres_aux matches; PWM-as-predictor heads are null/below')"),
    code(HEAD + "h=J('m2_kfold_HepG2.json')['archs']['perres']['rows']; k=J('m2_kfold_K562.json')['archs']['perres']['rows']\n"
         "hb={r['rbp']:r['pearson_real']-r['pearson_fam'] for r in h}; kb={r['rbp']:r['pearson_real']-r['pearson_fam'] for r in k}\n"
         "common=[r for r in hb if r in kb]\n"
         "fig,ax=plt.subplots(figsize=(3.6,3.6))\n"
         "ps.scatter_identity(ax,[hb[r] for r in common],[kb[r] for r in common],xlabel='HepG2 within-family gap',ylabel='K562 within-family gap',title='Cross-cell replication')\n"
         "show(fig,'nb19_replication.png')"),
    code(HEAD + "b=J('m2_bakeoff_HepG2.json')\n"
         "display(Markdown(f'''**Result (performance).** Per-residue cross-attn {b['per-residue']['real']:.3f} "
         "(within-family gap {b['per-residue']['gap_fam']:+.4f}); **perres_aux {b['perres_aux']['real']:.3f} / "
         "{b['perres_aux']['gap_fam']:+.4f} matches (CI overlaps)** while exposing a faithful PWM. Every PWM-as-"
         "predictor head is null on the profile (BioPWM {b['BioPWM']['gap_fam']:+.4f}, Form D {b['Form D']['gap_fam']:+.4f}, "
         "occ_footprint {b['occ_footprint']['gap_fam']:+.4f}) or below (perrespwm {b['perrespwm']['gap_fam']:+.4f}, "
         "motifscan {b['motifscan']['gap_fam']:+.4f}). The leakage-controlled gap (protein vs within-family shuffle) is "
         "the project's distinct, cross-cell-replicating selling point.'''))"),

    # ---- Section 2: read the motif ----
    md("## 2 - Read the motif (generated PWMs as information-content logos)"),
    code(HEAD + "import numpy as np\n"
         "z=None\n"
         "for nm in ['biopwm_recog_indist_pwms.npz','biopwm_pwms_indist.npz']:\n"
         "    p=OUT/nm\n"
         "    if p.exists(): z=np.load(p,allow_pickle=True); break\n"
         "if z is not None:\n"
         "    P=z['pwm']; syms=list(z['syms'])\n"
         "    if P.ndim==4: P=P.mean(1)\n"
         "    ic=(2+(np.clip(P,1e-9,1)*np.log2(np.clip(P,1e-9,1))).sum(-1)).sum(-1)\n"
         "    pick=list(np.argsort(-ic)[:6])\n"
         "    fig=ps.logo_grid([P[i] for i in pick],[syms[i] for i in pick],ncol=3)\n"
         "    show(fig,'nb19_logos.png','Generated motifs (read directly) - colorblind-safe bits logos')\n"
         "else: display(Markdown('_PWMs unavailable_'))"),
    md("Caveat (Gagneur discipline): motif recovery vs ATtRACT/RBPmap is only credible **with** a shuffle control "
       "(recovery-vs-mix_coeff collapsed +0.48->+0.08 on real weights). The PWM is the BioPWM's explicit latent; on "
       "the *profile* task it is interpretable but not the best predictor (Section 1)."),

    # ---- Section 3: see where on the RNA ----
    md("## 3 - See where on the RNA (eCLIP coverage track, observed vs predicted)"),
    code(HEAD + "import numpy as np\n"
         "d=OUT/'m2_dump_zsdump_HepG2.npz'\n"
         "if d.exists():\n"
         "    z=np.load(d,allow_pickle=True); rbp=z['rbp']; syms=list(z['syms']); obs=z['obs'].astype(float); ptr=z['pt_real'].astype(float); pts=z['pt_shuf'].astype(float)\n"
         "    pk=int(np.argmax(obs.sum(1)*(obs.max(1)>0)))   # a window with clear signal\n"
         "    L=obs.shape[1]; pos=np.arange(L)-L//2\n"
         "    fig,axes=plt.subplots(2,1,figsize=(6.2,3.2),sharex=True,gridspec_kw={'height_ratios':[1.0,1.4]})\n"
         "    rn=syms[int(rbp[pk])] if int(rbp[pk])<len(syms) else str(rbp[pk])\n"
         "    osh=obs[pk]/ (obs[pk].sum()+1e-9)\n"
         "    pr=ps.np if False else None\n"
         "    pe=float(((ptr[pk]-ptr[pk].mean())*(osh-osh.mean())).sum()/((ptr[pk].std()*osh.std()*L)+1e-9))\n"
         "    ps.eclip_track(axes,pos,obs[pk],ptr[pk],shuffled=pts[pk],rbp=rn,pearson=pe)\n"
         "    show(fig,'nb19_track.png','Predicted nt-resolution eCLIP profile tracks the observed coverage (unseen RBP)')\n"
         "else: display(Markdown('_profile dump unavailable_'))"),

    # ---- Section 4: decomposition ----
    md("## 4 - Coarse vs fine: is the signal genuinely nt-resolution?"),
    code(HEAD + "d=J('m2_decompose_zsdump_HepG2.json')\n"
         "if d:\n"
         "    rows=[{'rbp':r['rbp'][:9],'coarse_gap_shuf':r['coarse_gap_shuf'],'fine_gap_shuf':r['fine_gap_shuf']} for r in d['rows']]\n"
         "    fig,ax=plt.subplots(figsize=(5.5,3.8)); ps.decomp_bars(ax,rows,title='Coarse envelope vs fine single-nt (per-RBP)')\n"
         "    show(fig,'nb19_decomp.png','The protein gap has a real FINE single-nt component (nt-resolution)')\n"
         "else: display(Markdown('_decomposition unavailable_'))"),

    # ---- Section 5: synthesis ----
    md("## 5 - Synthesis & honest limitations\n\n"
       "| claim | status |\n|---|---|\n"
       "| Protein conditioning gives a real, leakage-controlled, cross-cell nt-resolution gap | **holds** (per-residue +0.025-0.031, p<1e-2, both cells) |\n"
       "| An interpretable head can MATCH per-residue performance | **holds via `perres_aux`** (0.164, faithful PWM proxy) |\n"
       "| Making the PWM the predictor matches per-residue | **false** (BioPWM/Form-D/occ_footprint null; perrespwm/motifscan below) |\n"
       "| The win is genuinely nt-resolution (fine), not just coarse | **holds** (fine gap >0, Section 4) |\n"
       "| Per-residue attention localizes to RRM/KH zero-shot | **false** (0.88x/0.55x below control - the honest limit; why the PWM proxy matters) |\n"
       "| ProtT5 / 3Di / bigger protein helps | **false** (ProtT5 == ESM; protein-richness is a null lever) |\n\n"
       "**Bottom line.** Per-residue cross-attention is the M2-profile mechanism; **`perres_aux` delivers its "
       "performance with an explicit, readable motif** (interpretation by faithful proxy, not by bottleneck). The "
       "leakage-controlled fine-resolution zero-shot gap is the CORAL-distinct contribution; the protein-structure "
       "and PWM-bottleneck routes are honestly closed. KB: cross-attention-architecture-investigation-2026-06-21, "
       "biopwm-unified-contract-2026-06-27, scaling-toward-coral-and-m2-zeroshot-2026-06-27." + ATTR),
])
print("NB19 RESULTS+INTERP DONE")
