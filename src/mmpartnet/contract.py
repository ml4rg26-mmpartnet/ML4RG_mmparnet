"""The ONE internal data type = the lab's ParnetDataElement, plus sparse<->dense
converters. Single source of truth = the lab scaffold's parnet_demo_utils.sparse_utils;
we import it if the scaffold is present, else fall back to vendored copies (the
converters are 4 stable torch one-liners) so the pipeline is robust to the tmp/ refs
being cleaned.

Contract (from sparse_utils.py):
  ParnetDataElement = {inputs: {sequence: str}, outputs: {task: SparseTensorDict}, meta: dict}
  SparseTensorDict  = {indices:(2,nnz), values:(nnz,), size:(N_TRACKS, L)}
"""
from __future__ import annotations
import sys
from typing import TypedDict
import torch

from . import config

_USING_LAB = False
try:  # prefer the lab's single source of truth
    if str(config.LAB_UTILS_SRC) not in sys.path:
        sys.path.insert(0, str(config.LAB_UTILS_SRC))
    from parnet_demo_utils.sparse_utils import (  # type: ignore
        ParnetDataElement, SparseTensorDict,
        torch_sparse_to_dense, torch_dense_to_sparse, dense_to_sparse_lists,
    )
    _USING_LAB = True
except Exception:  # vendored fallback (identical semantics)
    class SparseTensorDict(TypedDict):
        indices: torch.Tensor
        values: torch.Tensor
        size: torch.Size

    class _Inputs(TypedDict):
        sequence: str

    class ParnetDataElement(TypedDict):
        inputs: _Inputs
        outputs: dict
        meta: dict

    def torch_sparse_to_dense(sparse):
        return torch.sparse_coo_tensor(sparse["indices"], sparse["values"], sparse["size"]).to_dense()

    def torch_dense_to_sparse(dense):
        idx = dense.nonzero(as_tuple=False).T
        return {"indices": idx, "values": dense[idx[0], idx[1]], "size": dense.shape}

    def dense_to_sparse_lists(tensor):
        sp = tensor.to_sparse().coalesce()
        return {"indices": sp.indices().tolist(), "values": sp.values().tolist(), "size": list(tensor.shape)}


def make_element(sequence: str, outputs: dict, **meta) -> "ParnetDataElement":
    """Build a ParnetDataElement from a window sequence + {task: dense-or-sparse} outputs."""
    out = {}
    for k, v in outputs.items():
        out[k] = v if isinstance(v, dict) else torch_dense_to_sparse(
            v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32))
    return {"inputs": {"sequence": sequence}, "outputs": out, "meta": dict(meta)}


def validate_element(el) -> bool:
    """Schema check used by adapter tests."""
    if not (isinstance(el, dict) and {"inputs", "outputs", "meta"} <= set(el)):
        return False
    if not isinstance(el["inputs"].get("sequence"), str):
        return False
    for sp in el["outputs"].values():
        if not {"indices", "values", "size"} <= set(sp):
            return False
    return True


def using_lab_source() -> bool:
    return _USING_LAB
