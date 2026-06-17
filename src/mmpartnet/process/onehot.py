"""Canonical one-hot encoding (consolidates eclip_m1_gpu.onehot + rna_structure_gate.oh_str).
PARNET expects channels-first (4, L) over alphabet ACGT (U->T); non-ACGT -> zero column."""
from __future__ import annotations
import numpy as np
import torch

_BIDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def onehot(seq: str) -> torch.Tensor:
    """seq -> float tensor (4, L), channels-first."""
    s = seq.upper()
    arr = np.zeros((4, len(s)), dtype=np.float32)
    for j, ch in enumerate(s):
        i = _BIDX.get(ch)
        if i is not None:
            arr[i, j] = 1.0
    return torch.from_numpy(arr)


def batch_onehot(seqs, device=None) -> torch.Tensor:
    """list[str] of equal length -> (B, 4, L)."""
    t = torch.stack([onehot(s) for s in seqs])
    return t.to(device) if device else t
