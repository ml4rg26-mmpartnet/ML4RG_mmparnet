"""Conditioning-head registry - the single plug-in seam for a teammate model.

Any protein-conditioned head registers a NAME here; ``RunConfig.conditioning`` selects it and
``eval/protocol.py`` runs it through the SAME leakage-controlled, family-disjoint evaluation. To add a
model: drop a module in ``models/``, implement a head mapping (frozen PARNET RNA features, protein rep)
-> a binding logit and/or a per-nt profile, then add ONE ``REGISTRY`` row (lazy import -> no new hard
deps for unused heads).

Heads are heterogeneous BY DESIGN - pooled-binary (M1) vs per-residue-profile (M2) - so a single
forward signature would be a false uniformity. Each row records ``task`` (binary|profile) and ``inputs``
(pooled|perres) so the harness/adapter feeds the right tensors. ``build_head(name)`` does the lazy
import and returns the class; instantiate it with the dims your run needs.

The cross-attention A/B (``xattn`` = dgu, ``xattn2`` = dfra) is intentionally two coexisting variants
behind this one seam - benchmark on a leave-out-RBP split with the mandatory controls, keep the winner
as default, and drop the loser once the comparison notebook captures it.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class HeadSpec:
    name: str
    module: str        # dotted, relative to mmpartnet.models
    cls: str
    task: str          # "binary" | "profile"
    inputs: str        # "pooled" | "perres"
    owner: str
    note: str


REGISTRY: dict[str, HeadSpec] = {
    "early": HeadSpec(
        "early", "early_fusion", "EarlyFusion", "binary", "pooled", "cgerards",
        "concat[protein, pooled PARNET body] -> MLP (the mechanism-ladder floor / early-fusion baseline)."),
    "film": HeadSpec(
        "film", "film", "ProteinCellFiLMProfileHead", "profile", "perres", "dgu",
        "per-nt FiLM(gamma,beta from protein+cell) over PARNET features -> additive target/control mix (multitask)."),
    "xattn": HeadSpec(
        "xattn", "cross_attention_dgu", "ProteinCellCrossAttentionProfileHead", "profile", "perres", "dgu",
        "cell-FiLM bidirectional cross-attention + latent protein compressor (A/B variant A vs xattn2)."),
    "xattn2": HeadSpec(
        "xattn2", "cross_attention_dfra", "TFBindCrossAttentionProfileHead", "profile", "perres", "dfra",
        "residue-level ProtT5 cross-attention + position-weighted pool (A/B variant B vs xattn)."),
    "conditioned": HeadSpec(
        "conditioned", "heads", "ConditionedHead", "binary", "pooled", "ours",
        "FiLM-modulated pooled PARNET features -> binding logit (M1 binary; the leave-out-RBP gate head)."),
    "perres": HeadSpec(
        "perres", "cross_attn_head", "CrossAttnHead", "profile", "perres", "ours",
        "per-residue protein cross-attention over PARNET features -> profile (our M2 winner)."),
    "perres_bidir": HeadSpec(
        "perres_bidir", "cross_attn_head", "BidirCrossAttnHead", "profile", "perres", "ours",
        "bidirectional per-residue cross-attention (ties per-residue in-distribution)."),
}


def list_heads() -> list[str]:
    return sorted(REGISTRY)


def head_spec(name: str) -> HeadSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown conditioning head {name!r}; registered: {list_heads()}")
    return REGISTRY[name]


def build_head(name: str):
    """Import + return the head CLASS registered under ``name`` (lazy - unused heads cost no import).
    Instantiate with the dims your run needs, e.g. ``build_head('early')(dr=512, dp=1024)``."""
    spec = head_spec(name)
    mod = import_module(f"mmpartnet.models.{spec.module}")
    return getattr(mod, spec.cls)
