"""mmpartnet.protein: swappable per-RBP protein representation (CONTRACT.md swap-in #2).

    from mmpartnet.protein import get_protein
    rep = get_protein("ribex_proxy")          # esm650_pooled | ribex_proxy | ribex_real
    e = rep.vector("QKI")                       # 1-D float32, or None
    prot = rep.map(["QKI", "PTBP1"])            # {rbp: vector} for the FiLM head

Flip `ribex_proxy` -> `ribex_real` (lab-trained RIBEX) with one name change, zero downstream edits.
"""
from __future__ import annotations
from .base import ProteinConfig, ProteinSource
from .registry import get_protein, list_proteins, register

__all__ = ["ProteinConfig", "ProteinSource", "get_protein", "list_proteins", "register"]
