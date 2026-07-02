"""Models - the frozen pretrained PARNET base.

PARNET is the lab's PRETRAINED RNA encoder, loaded FROZEN: we only read its body features
(``parnet.ParnetModel.body_feats``, under ``torch.no_grad``) and its per-nt profile head. The protein
side (ESM2 / RIBEX) lives in ``mmpartnet.protein``. ``load_parnet`` is weight-agnostic, so swapping the
checkpoint is a one-line config change (CONTRACT.md swap-in #1).
"""
from __future__ import annotations

from .early_fusion import EarlyFusionConcatHead
from .film import ProteinCellFiLMProfileHead
from .parnet import ParnetModel, load_parnet
from .registry import REGISTRY as HEAD_REGISTRY, HeadSpec, list_heads, head_spec, build_head
# head classes are torch-only imports (no data deps) — export the common ones so teammate modules can
# `from mmpartnet.models import ProteinCellFiLMProfileHead` etc.; use build_head(name) for lazy access.
from .heads import ConditionedHead
from .early_fusion import EarlyFusion, fit_predict
from .film import ProteinCellFiLMProfileHead

__all__ = ["EarlyFusionConcatHead", "ParnetModel", "load_parnet",
           "HEAD_REGISTRY", "HeadSpec", "list_heads", "head_spec", "build_head",
           "ConditionedHead", "EarlyFusion", "fit_predict", "ProteinCellFiLMProfileHead"]
