"""Decomposition — separate what the protein-conditioning REALLY buys from RBP-identity lookup.

Two decompositions:

1. Profile-shape decomposition (M2): ``shp_pearson`` / ``components`` (re-exported from
   experiments.m2_decompose) split a profile's agreement into a smooth (region) part and a sharp
   (single-nt) part, so "conditioning improved the profile" is attributed to the right length scale.

2. Regime decomposition (the headline): ``regime_table`` lays the metric out over
   {in-distribution, zero-shot} x {binary, profile, affinity}. The finding it encodes:
     - IN-DISTRIBUTION the protein signal is large but is an RBP-IDENTITY LOOKUP (protein-shuffle
       collapses it) -> not biology, not transferable;
     - ZERO-SHOT on a clean leave-out-RBP backbone it is ~null at binary, small-but-real at profile,
       real-and-growing at affinity.
   Feed it the numbers you computed (e.g. from held_rbp_gate seen/unseen + m2_profile + affinity grid).
"""
from __future__ import annotations

# reusable profile-shape decomposition primitives (single source of truth)
from mmpartnet.experiments.m2_decompose import gsmooth, shp_pearson, components  # noqa: F401


def regime_table(binary=None, profile=None, affinity=None):
    """Assemble the {in_dist, zero_shot} x {binary, profile, affinity} summary.

    Each argument is a dict ``{"in_dist": <value>, "zero_shot": <value>, "shuffle": <value?>}`` (any
    subset). Returns a nested dict + a ``verdict`` per axis flagging whether the zero-shot signal
    survives (>0 and not explained by shuffle). This is the L1 decomposition notebook's data object."""
    axes = {"binary": binary, "profile": profile, "affinity": affinity}
    out = {}
    for axis, d in axes.items():
        if not d:
            continue
        zs = d.get("zero_shot")
        idd = d.get("in_dist")
        shuf = d.get("shuffle")
        survives = (zs is not None and zs == zs and zs > (shuf if shuf is not None else 0.0))
        out[axis] = {
            "in_dist": idd, "zero_shot": zs, "shuffle": shuf,
            "identity_lookup": (idd is not None and shuf is not None and (idd - shuf) > 0.05),
            "zero_shot_survives": bool(survives),
            "verdict": ("zero-shot signal survives" if survives else
                        "zero-shot ~null (in-dist was identity lookup)"),
        }
    return out
