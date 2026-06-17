"""POPULATED proxy: ESM2-650 (+) STRING-PE (704-d). NOT faithful RIBEX (concat not FiLM, reduced
PE not full PPR, frozen ESM not LoRA) -> every number from this rep is a SURROGATE, mark it 'proxy'.
This is what the initial real run uses until ribex_real lands. Delegates to models.ribex.ribex_vector.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from ...models import ribex
from ..base import ProteinSource
from ..registry import register


@register("ribex_proxy")
class RibexProxy(ProteinSource):
    def vector(self, rbp: str) -> Optional[np.ndarray]:
        return ribex.ribex_vector(rbp, mode="ribex_proxy")
