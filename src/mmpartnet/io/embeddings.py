"""ProteinRep — per-RBP protein representations (the modality that conditions PARNET).
Wraps the local npz assets; keyed by RBP symbol. Protein reps live OUTSIDE ParnetDataElement
(they are per-RBP, not per-window) and are joined at batch assembly.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np

from .. import config


@lru_cache(maxsize=1)
def _load(npz_path):
    return dict(np.load(npz_path, allow_pickle=True)) if npz_path.exists() else {}


class ProteinRep:
    """Access pooled ESM-650 (640-d), STRING PE (64-d), per-residue, and ortholog embeddings."""

    def __init__(self):
        self.pooled = _load(config.EMB_POOLED)        # gene -> (640,)
        self.pe_string = _load(config.PE_STRING)      # gene -> (64,)
        self.perres = None                            # lazy (39 MB)
        self.xspecies = None

    def genes(self):
        return sorted(self.pooled.keys())

    def esm(self, symbol):
        v = self.pooled.get(symbol)
        return None if v is None else np.asarray(v, np.float32).ravel()

    def string_pe(self, symbol):
        v = self.pe_string.get(symbol)
        return None if v is None else np.asarray(v, np.float32).ravel()

    def esm_string(self, symbol):
        """Concatenated ESM ⊕ STRING-PE proxy (the 'ribex_proxy' input; FiLM is applied downstream)."""
        e = self.esm(symbol)
        if e is None:
            return None
        s = self.string_pe(symbol)
        return e if s is None else np.concatenate([e, s])

    def per_residue(self, symbol):
        if self.perres is None:
            self.perres = _load(config.EMB_PERRES)
        v = self.perres.get(symbol)
        return None if v is None else np.asarray(v, np.float32)

    def ortholog(self, symbol, species):
        if self.xspecies is None:
            self.xspecies = _load(config.EMB_XSPECIES)
        v = self.xspecies.get(f"{symbol}|{species}")
        return None if v is None else np.asarray(v, np.float32).ravel()
