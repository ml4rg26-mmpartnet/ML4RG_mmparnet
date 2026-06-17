"""STUB (drop-in): lab-trained RIBEX fused embedding. Faithful protein rep (CONTRACT.md swap #2).

Becomes live the moment an .npz of {symbol: vector} exists: set env ML4RG_RIBEX (or cfg.extra
['ribex_npz']). Two credible routes to obtain it (both lab/ml4rg-associated): ask a supervisor for the
trained checkpoint + run RIBEX/scripts/model_inference.py, OR retrain marsico-lab/RIBEX. The resolver
(models.ribex.ribex_vector mode='ribex_real') already raises a precise instruction if it is missing.
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np

from ...models import ribex
from ..base import ProteinSource
from ..registry import register


@register("ribex_real")
class RibexReal(ProteinSource):
    def vector(self, rbp: str) -> Optional[np.ndarray]:
        # allow cfg.extra['ribex_npz'] to set the path (else env ML4RG_RIBEX, handled by the resolver)
        npz = self.cfg.extra.get("ribex_npz")
        if npz:
            os.environ.setdefault("ML4RG_RIBEX", str(npz))
        return ribex.ribex_vector(rbp, mode="ribex_real")  # raises FileNotFoundError w/ instructions if unset
