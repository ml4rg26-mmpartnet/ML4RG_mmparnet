# `mmpartnet.protein` — swappable protein rep (swap-in #2)

Per-RBP conditioning vector behind one interface; mirrors `mmpartnet.data`.

```python
from mmpartnet.protein import get_protein, list_proteins
rep = get_protein("ribex_proxy")     # esm650_pooled | ribex_proxy | ribex_real
rep.vector("QKI")                    # 1-D float32 or None
rep.map(["QKI", "PTBP1"])            # {rbp: vector} for the FiLM head
```

| name | status | what |
|------|--------|------|
| `esm650_pooled` | populated | ESM2-650 pooled, 640-d (conservative baseline) |
| `ribex_proxy` | populated | ESM2-650 (+) STRING-PE, 704-d (SURROGATE; mark numbers 'proxy') |
| `ribex_real` | stub | lab-trained RIBEX fused embedding (env `ML4RG_RIBEX`, lab-gated) |

Bodies delegate to `models.ribex.ribex_vector` (single source of truth). Add a rep = drop a module in
`providers/`, `@register("name")`, import it in `providers/__init__.py`.
