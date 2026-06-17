# `mmpartnet.splits` — swappable split axis (swap-in #4)

The split axis IS the generalization claim. Mirrors `mmpartnet.data`.

```python
from mmpartnet.splits import SplitConfig, get_split, holdout_chrom
get_split(rbps, SplitConfig(axis="family", held_family="RRM"))   # -> RbpSplit(train, test)
holdout_chrom(windows, "chr1")                                    # naive window-level holdout
```

| axis | status | measures |
|------|--------|----------|
| `naive` | populated | in-distribution profile fit (window-level chrom holdout) |
| `family` | populated | cross-family transfer (leave-one-ATtRACT-family-out) |
| `rbp_holdout` | functional | clean zero-shot held-RBP (the decisive M2 axis) |

`rbp_holdout` is only an HONEST zero-shot test on a leave-out-pretrained PARNET (`m2.leaveout_parnet`,
swap-in #1, lab-gated). Add an axis = drop a module in `strategies/`, `@register`, import it.
