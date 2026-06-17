"""POPULATED: ESM2-650 pooled (640-d) protein rep. Delegates to models.ribex.ribex_vector."""
from __future__ import annotations
from typing import Optional
import numpy as np

from ...models import ribex
from ..base import ProteinSource
from ..registry import register


@register("esm650_pooled")
class Esm650Pooled(ProteinSource):
    def vector(self, rbp: str) -> Optional[np.ndarray]:
        return ribex.ribex_vector(rbp, mode="esm650_pooled")
