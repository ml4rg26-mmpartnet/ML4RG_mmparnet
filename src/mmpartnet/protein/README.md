# `mmpartnet.protein` — swappable protein rep (swap-in #2)

Per-RBP conditioning vector behind one interface; mirrors `mmpartnet.data`.

```python
from mmpartnet.protein import get_protein, list_proteins
rep = get_protein("ribex_proxy")     # esm650_pooled | ribex_proxy | ribex_real | prott5_h5
rep.vector("QKI")                    # 1-D float32 or None
rep.map(["QKI", "PTBP1"])            # {rbp: vector} for the FiLM head
```

| name | status | what |
|------|--------|------|
| `esm650_pooled` | populated | ESM2-650 pooled, 640-d (conservative baseline) |
| `ribex_proxy` | populated | ESM2-650 (+) STRING-PE, 704-d (SURROGATE; mark numbers 'proxy') |
| `ribex_real` | stub | lab-trained RIBEX fused embedding (env `ML4RG_RIBEX`, lab-gated) |
| `prott5_h5` | populated | pooled ProtT5 vectors from the shared H5 file, used by `dgu/film-multitask` |

Bodies delegate to `models.ribex.ribex_vector` (single source of truth). Add a rep = drop a module in
`providers/`, `@register("name")`, import it in `providers/__init__.py`.

## FiLM branch note

The FiLM multitask branch uses `prott5_h5` through this same registry instead
of opening the ProtT5 H5 file directly inside the training loop. The current
track-level workflow resolves embeddings by the exact H5 key in
`mmpartnet_out/prott5_track_map.tsv`; the provider also supports
`vector("QKI")` for the standard RBP-symbol interface.
