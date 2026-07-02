# Modular conditioning heads - the plug-in seam

This base adds the shared infrastructure that lets every protein-conditioning model plug into ONE
leakage-controlled, family-disjoint evaluation. Add a head = drop a file + one registry row; you never
touch the harness.

## The three pieces

| Piece | File | What it gives you |
|-------|------|-------------------|
| Head registry | `models/registry.py` | `name -> HeadSpec`; `list_heads()`, `head_spec(name)`, `build_head(name)` (lazy import - unused heads cost nothing) |
| Config selector + honesty gate | `config.py` | `RunConfig.conditioning` (which head) + `RunConfig.control` (which nulls); `honest_zero_shot()` |
| Eval contract | `eval/` | `held_rbp_gate`, `run_controls`, `family_disjoint_assert`, metrics (`roc_auc`, `average_precision`, `profile_pearson`) |

## Registered heads

| key | module | task | owner | note |
|-----|--------|------|-------|------|
| `early` | `models/early_fusion.py` | binary | cgerards | concat[protein, pooled PARNET body] -> MLP (baseline floor) |
| `film` | `models/film.py` | profile | dgu | per-nt FiLM(gamma,beta from protein+cell) over PARNET features (multitask) |
| `xattn` | `models/cross_attention_dgu.py` | profile | dgu | cell-FiLM bidirectional cross-attention + latent protein compressor (A/B variant A) |
| `xattn2` | `models/cross_attention_dfra.py` | profile | dfra | residue-level ProtT5 cross-attention + position-weighted pool (A/B variant B) |

`xattn` and `xattn2` coexist on purpose: the A/B winner is decided on a leave-out-RBP split WITH the
mandatory controls; keep both until a comparison captures it, then drop the loser.

## Add your own head

```python
# 1. models/my_head.py
import torch.nn as nn
class MyHead(nn.Module):
    def forward(self, rna_feats, protein_rep):   # rna_feats: frozen PARNET body features; protein_rep: your rep
        ...                                        # return a binding logit and/or a per-nt profile

# 2. one row in models/registry.py
"myhead": HeadSpec("myhead", "my_head", "MyHead", "profile", "perres", "you", "one-line description"),

# 3. select it
RunConfig(conditioning="myhead")
build_head("myhead")(dr=512, dp=1024)
```

## Evaluate any head the same way

```python
from mmpartnet.eval import held_rbp_gate, torch_head_scorer, family_disjoint_assert
family_disjoint_assert(train_names, held_names, family_of)     # split-level leakage guarantee
res = held_rbp_gate(torch_head_scorer(head), feats, held_k, prot_emb, te_y, ti_keep, syms, fam_lab)
# res.controls -> protein-shuffle / within-family, each B3 fire-checked (a control that does not move
# the metric is FLAGGED, not a silent pass). res.honest_zero_shot -> False on the leaked all-223 body.
```

## Honesty gate

`config.honest_zero_shot()` returns `False` on the default all-223 PARNET checkpoint (it LEAKS the held
RBP into the frozen body). Every zero-shot / multimodal number computed on it is proxy-level. Point
`ML4RG_PARNET_WEIGHTS` at a leave-out checkpoint (+ `ML4RG_HONEST_ZEROSHOT=1`) to promote a run to a
headline claim.

## Splits

`splits/strategies/` adds `paralog` (leave-one-paralog-group-out, the within-well transfer control)
alongside `naive` / `family` / `rbp_holdout`. Select via `SplitConfig(axis="paralog")`.
