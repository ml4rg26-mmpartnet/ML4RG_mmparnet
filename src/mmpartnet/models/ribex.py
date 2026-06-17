"""RIBEX — the protein representation that conditions PARNET. The three states + the resolver.

WHAT RIBEX IS (Marsico lab; Firmani et al., bioRxiv 2026.03.13.711639): ESM2-650M (LoRA-adapted)
sequence embedding, FiLM-conditioned by a STRING v12 PPI **personalized-PageRank** positional
encoding. It is an RBP-vs-non-RBP identification model; the reusable signal for us is the fused
per-RBP embedding.

THE PROBLEM: no RIBEX weights are released (the RIBEX repo (marsico-lab/RIBEX) is code + pipeline.sh
only). So we resolve the protein rep in three states:

  mode='ribex_proxy'  (AVAILABLE NOW, on disk) — ESM2-650 pooled (640-d) (+) STRING-PE (64-d) from
      io.embeddings.ProteinRep (embeddings_all.npz + pe_string.npz). A 640/704-d proxy. NOT faithful
      (concat not FiLM, reduced-PE not full PPR, frozen ESM not LoRA) -> mark every number 'proxy'.
      This is what the initial real run uses.

  mode='esm650_pooled' — ESM2-650 pooled only (640-d). The conservative baseline protein rep.

  mode='ribex_real'  (DROP-IN when it lands) — set env ML4RG_RIBEX to an .npz of {symbol: vector}.
      Two credible, lab/ml4rg-associated ways to obtain it:
        (A) ASK MOYON for the trained RIBEX checkpoint, then run RIBEX's scripts/model_inference.py
            (the RIBEX repo) to harvest the post-FiLM fused embedding -> save as the npz.
            Fastest; it is the lab's own weights.
        (B) RETRAIN from the public marsico-lab/RIBEX: build RIC/Bressin19/InterPro tables -> ESM2-650
            embeddings -> STRING v12 PPR PE (9606.protein.links.full.v12.0) -> LoRA + FiLM-PE random
            search (README + pipeline.sh). Reproducible, compute-heavy (HPC/conda).
The conditioning head (a FiLM-conditioned head) is rep-agnostic: only the vector source changes,
so flipping the mode is a one-line config swap with zero downstream edits (CONTRACT.md swap #2).
"""
from __future__ import annotations
import os
import numpy as np

from ..io.embeddings import ProteinRep

VALID = ("ribex_proxy", "esm650_string", "esm650_pooled", "esm", "ribex_real")


def ribex_vector(symbol: str, mode: str = "ribex_proxy", rep: "ProteinRep | None" = None):
    """Resolve the protein conditioning vector for ``symbol`` under ``mode``. Returns a 1-D float32
    np.ndarray, or None if the symbol is absent for that mode."""
    rep = rep or ProteinRep()
    if mode == "ribex_real":
        path = os.environ.get("ML4RG_RIBEX")
        if path and os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            return np.asarray(d[symbol], np.float32).ravel() if symbol in d else None
        raise FileNotFoundError(
            "ribex_real requested but ML4RG_RIBEX (an .npz of {symbol: vector}) is unset/missing. "
            "Obtain real RIBEX (ask a supervisor for the checkpoint + run RIBEX/scripts/model_inference.py, "
            "or retrain marsico-lab/RIBEX), or fall back to mode='ribex_proxy'. See module docstring.")
    if mode in ("ribex_proxy", "esm650_string"):
        return rep.esm_string(symbol)              # ESM2-650 (+) STRING-PE
    if mode in ("esm650_pooled", "esm"):
        return rep.esm(symbol)
    raise ValueError(f"unknown protein mode {mode!r} (valid: {VALID})")


def protein_map(symbols, mode: str = "ribex_proxy", rep: "ProteinRep | None" = None) -> dict:
    """{symbol: vector} for the symbols resolvable under ``mode`` (skips missing). Convenience for
    the experiments: ``prot = ribex.protein_map(rbps, mode=cfg.protein)``."""
    rep = rep or ProteinRep()
    out = {}
    for g in symbols:
        v = ribex_vector(g, mode=mode, rep=rep)
        if v is not None:
            out[g] = v
    return out
