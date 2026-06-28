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

__all__ = ["EarlyFusionConcatHead", "ParnetModel", "ProteinCellFiLMProfileHead", "load_parnet"]
