"""Models - the frozen pretrained PARNET base.

PARNET is the lab's PRETRAINED RNA encoder, loaded FROZEN: we only read its body features
(``parnet.ParnetModel.body_feats``, under ``torch.no_grad``) and its per-nt profile head. The protein
side (ESM2 / RIBEX) lives in ``mmpartnet.protein``. ``load_parnet`` is weight-agnostic, so swapping the
checkpoint is a one-line config change (CONTRACT.md swap-in #1).
"""
from __future__ import annotations

from .film import ProteinCellFiLMProfileHead
from .parnet import ParnetModel, load_parnet
from .early_fusion import EarlyFusion
from .heads import ConditionedHead
# the plug-in seam; head classes load lazily via build_head(name) so unused heads cost no import
from .registry import REGISTRY as HEAD_REGISTRY, HeadSpec, list_heads, head_spec, build_head

__all__ = ["ParnetModel", "ProteinCellFiLMProfileHead", "load_parnet", "EarlyFusion", "ConditionedHead",
           "HEAD_REGISTRY", "HeadSpec", "list_heads", "head_spec", "build_head"]
