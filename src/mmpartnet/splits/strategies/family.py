"""POPULATED: family axis (leave-one-family-out). Cross-family transfer test: train on all RBPs
whose ATtRACT family != held_family, test on the held family. Family map from io.cohort. If
cfg.held_family is empty, leave out the first family encountered (deterministic by sorted order)."""
from __future__ import annotations
from ..base import RbpSplit
from ..registry import register


def _family_map(rbps):
    """{rbp: family} via io.cohort.attract_families; falls back to '?' for unmapped (needs ATtRACT)."""
    try:
        from ...io import cohort
        fam = cohort.attract_families(list(rbps))
        return {g: (fam[g][0] if g in fam else "?") for g in rbps}
    except Exception:
        return {g: "?" for g in rbps}


@register("family")
def family(rbps, cfg, meta=None) -> RbpSplit:
    fmap = (meta or {}).get("family") if meta else None
    fmap = fmap or _family_map(rbps)
    held = cfg.held_family or next((fmap[g] for g in sorted(rbps) if fmap.get(g, "?") != "?"), "?")
    test = tuple(g for g in rbps if fmap.get(g, "?") == held)
    train = tuple(g for g in rbps if g not in test)
    return RbpSplit(train=train, test=test, axis="family",
                    note=f"leave-out family={held!r} (n_test={len(test)})")
