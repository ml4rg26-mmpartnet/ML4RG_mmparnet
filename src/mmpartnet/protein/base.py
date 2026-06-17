"""Protein-representation layer (CONTRACT.md swap-in #2), same shape as `mmpartnet.data`.

A `ProteinSource` maps an RBP symbol -> a 1-D conditioning vector. Swap the rep by registry name
(`esm650_pooled` / `ribex_proxy` / `ribex_real`) with zero downstream edits; the FiLM head
(a FiLM-conditioned head) is rep-agnostic. Bodies delegate to the existing resolver
(`models.ribex.ribex_vector`) so this is a thin, uniform facade, not a second implementation.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ProteinConfig:
    """Which protein rep to use. `mode` = registry name; `extra` carries per-provider knobs
    (e.g. {'ribex_npz': path} for ribex_real)."""
    mode: str = "ribex_proxy"
    extra: dict = field(default_factory=dict)


class ProteinSource(ABC):
    """Swappable per-RBP protein vector source. Implement `vector`; `dim`/`map`/`available` derive."""

    name: str = "ribex_proxy"

    def __init__(self, cfg: "ProteinConfig | None" = None):
        self.cfg = cfg or ProteinConfig(mode=getattr(self, "name", "ribex_proxy"))

    @abstractmethod
    def vector(self, rbp: str) -> Optional[np.ndarray]:
        """1-D float32 conditioning vector for `rbp`, or None if unavailable for this rep."""

    @property
    def dim(self) -> Optional[int]:
        """Vector length (probed from the first available rep); None if nothing resolvable."""
        d = getattr(self, "_dim", "unset")
        if d != "unset":
            return d
        self._dim = None
        for g in (self.cfg.extra.get("probe") or ["QKI", "PTBP1", "IGF2BP1"]):
            v = self.vector(g)
            if v is not None:
                self._dim = int(np.asarray(v).ravel().shape[0]); break
        return self._dim

    def map(self, rbps) -> dict:
        """{rbp: vector} for the resolvable symbols (skips missing)."""
        out = {}
        for g in rbps:
            v = self.vector(g)
            if v is not None:
                out[g] = np.asarray(v, np.float32).ravel()
        return out

    def available(self, rbps) -> list:
        return [g for g in rbps if self.vector(g) is not None]
