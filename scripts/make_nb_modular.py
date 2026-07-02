"""Build the modular-eval + head-registry plug-in demo notebook (executed, synthetic data so it runs
from a clone with no lab data). This is the "how a teammate reuses our work" artifact.

  python scripts/make_nb_modular.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbgen import md, code, build  # noqa: E402

CELLS = [
    md("# Modular eval + head registry — the plug-in seam\n"
       "How to run **any** protein-conditioning head through the SAME leakage-controlled, "
       "family-disjoint evaluation. Synthetic data — runs from a clone, no lab assets. "
       "See `docs/MERGE_PLAN.md` §6 (shared interfaces) and `src/mmpartnet/eval/`."),

    md("## 1. The head registry\n"
       "`RunConfig.conditioning` selects a head by name; `build_head(name)` lazy-imports it. To add "
       "your model: drop a module in `models/`, implement a head, add one row in `models/registry.py`."),
    code("from mmpartnet.models import list_heads, head_spec, build_head\n"
         "for n in list_heads():\n"
         "    s = head_spec(n)\n"
         "    print(f'{n:14s} task={s.task:7s} inputs={s.inputs:6s} owner={s.owner:9s} {s.cls}')\n"
         "print()\n"
         "print('build_head(\"xattn\") ->', build_head('xattn').__module__)"),

    md("## 2. The leakage-controlled leave-out-RBP gate\n"
       "`held_rbp_gate(score_fn, ...)` is model-agnostic: give a `score_fn(feats, protein_vec)` and the "
       "held-out RBPs, get per-RBP AUROC + the mandatory protein-shuffle control, **fire-checked**. A "
       "protein-USING head must make the shuffle collapse (real > shuffle); a protein-IGNORING head does "
       "not — and the harness FLAGS that (B3 discipline: a silent control is a broken eval, not a win)."),
    code("import numpy as np\n"
         "from mmpartnet.eval import held_rbp_gate\n"
         "rng = np.random.default_rng(0)\n"
         "K, W = 6, 400                      # 6 held RBPs, 400 windows\n"
         "t = rng.integers(0, K, size=W)     # each window's latent 'bound-by' RBP\n"
         "feats = np.eye(K)[t]               # (W,K) window features (one-hot of the latent type)\n"
         "te_y  = np.eye(K)[t]               # (W,K) labels: window w is bound by RBP t[w]\n"
         "prot  = {k: np.eye(K)[k] for k in range(K)}   # protein rep = identity of RBP k\n"
         "ti_keep = list(range(K)); held_k = list(range(K))\n"
         "syms = [f'RBP{k}' for k in range(K)]; fam = [f'fam{k%3}' for k in range(K)]\n"
         "\n"
         "protein_using  = lambda F, pv: F @ pv            # uses the protein rep\n"
         "protein_blind  = lambda F, pv: F.sum(1)          # ignores the protein rep\n"
         "\n"
         "for name, fn in [('protein-USING', protein_using), ('protein-BLIND', protein_blind)]:\n"
         "    r = held_rbp_gate(fn, feats, held_k, prot, te_y, ti_keep, syms, fam, bar=0.65)\n"
         "    ps = r.controls['protein_shuffle']\n"
         "    print(f'{name:14s} real={r.real_mean:.2f} shuffle={ps[\"mean_auroc\"]:.2f} '\n"
         "          f'fired={ps[\"fired\"]} warn={ps[\"warn\"]}  honest_zero_shot={r.honest_zero_shot}')"),
    md("**Read-out:** the protein-USING head separates the RBPs (real AUROC ~1.0) and the protein-shuffle "
       "collapses it (control *fires*); the protein-BLIND head scores identically under shuffle, so the "
       "control does NOT fire and is flagged. `honest_zero_shot=False` because the default checkpoint is "
       "the leaked all-223 body — the number is proxy-level until leave-out weights are set."),

    md("## 3. The family-disjoint guarantee + splits\n"
       "`family_disjoint_assert` refuses a split where a family appears in both train and eval; the "
       "`paralog` axis is the within-well transfer control (leave-out-chromosome = the `naive` axis)."),
    code("from mmpartnet.eval import family_disjoint_assert\n"
         "from mmpartnet.splits.registry import list_splits, get_split\n"
         "from mmpartnet.splits.base import SplitConfig\n"
         "print('split axes:', list_splits())\n"
         "sp = get_split(['A','B','C','D'], SplitConfig(axis='paralog'),\n"
         "               meta={'paralog': {'A':'g1','B':'g1','C':'g2','D':'g2'}})\n"
         "print('paralog leave-out:', 'train', sp.train, 'test', sp.test)\n"
         "try:\n"
         "    family_disjoint_assert(['A','B'], ['C','A'], {'A':'f1','B':'f2','C':'f1'})\n"
         "except AssertionError as e:\n"
         "    print('caught:', str(e)[:70], '...')"),

    md("## 4. CORAL-comparable metrics + the affinity certificate\n"
       "`coral_f1_auroc(..., seen_mask=)` always reports the UNSEEN (cold-start) block — the real claim. "
       "`validate_grid` is the affinity far>near test with a family-block permutation null."),
    code("from mmpartnet.metrics import coral_f1_auroc, validate_grid\n"
         "pred = np.array([0.9,0.8,0.7,0.2,0.1,0.05]); y = np.array([1,1,1,0,0,0])\n"
         "seen = np.array([1,1,0,1,0,0], bool)\n"
         "r = coral_f1_auroc(pred, y, seen_mask=seen, best_thr=True)\n"
         "print('overall', r['overall']['f1'], r['overall']['auroc'], '| unseen n', r['unseen']['n'])\n"
         "vg = validate_grid(pred, np.array([.1,.2,.3,.9,1.,1.1]), n_perm=500, seed=0)\n"
         "print('affinity far>near effect', round(vg['effect'],3), 'p', round(vg['p'],4))"),

    md("## 5. Plug YOUR head in (recipe)\n"
       "```python\n"
       "# 1) models/your_head.py: implement forward(rna_feats, protein_rep) -> logit and/or profile\n"
       "# 2) models/registry.py: add HeadSpec('mine','your_head','YourHead','profile','perres','you', '...')\n"
       "# 3) RunConfig(conditioning='mine'); build_head('mine')(**dims)\n"
       "# 4) evaluate: held_rbp_gate(torch_head_scorer(head), feats, held_k, prot, te_y, ti_keep, syms, fam)\n"
       "#    -> same controls + family-disjoint guarantee everyone else is held to.\n"
       "```\n"
       "Set `ML4RG_PARNET_WEIGHTS` to a leave-out checkpoint (+ `ML4RG_HONEST_ZEROSHOT=1`) to promote a "
       "run from proxy to a real held-out claim."),
]

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    build(root / "notebooks" / "demo" / "05_modular_eval_and_registry.ipynb",
          root / "notebooks" / "demo" / "executed" / "05_modular_eval_and_registry_executed.ipynb",
          CELLS, timeout=300)
