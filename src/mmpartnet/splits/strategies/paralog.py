"""POPULATED: paralog axis (leave-one-paralog-group-out) — the family axis at PARALOG granularity.

This is the interpolation-in-RBP-space control: paralogs are the SAME "well" in protein space (near-
identical binding), so holding out a whole paralog group and testing on it measures whether the head
can transfer WITHIN a well (easy) vs the family axis (across wells). Contrast with an independent-family
split (scaling/family_curve): the finding is that paralog/within-well transfer is easy and does NOT
scale, whereas independent-family count is the lever. Paralog labels come from `meta['paralog']`
(rbp -> group id), or fall back to the ATtRACT family map.

Leave-out-chromosome is NOT here on purpose: it is a WINDOW-level split = the `naive` axis with
`SplitConfig.held_chrom` (+ `splits.base.holdout_chrom`), not an RBP-level partition.
"""
from __future__ import annotations

from ..base import RbpSplit
from ..registry import register


def _paralog_map(rbps, meta):
    pmap = (meta or {}).get("paralog") if meta else None
    if pmap:
        return {g: pmap.get(g, "?") for g in rbps}
    # fallback: ATtRACT family as a coarse paralog proxy
    try:
        from ...io import cohort
        fam = cohort.attract_families(list(rbps))
        return {g: (fam[g][0] if g in fam else "?") for g in rbps}
    except Exception:
        return {g: "?" for g in rbps}


@register("paralog")
def paralog(rbps, cfg, meta=None) -> RbpSplit:
    pmap = _paralog_map(rbps, meta)
    # held group: cfg.held_family reused as the group selector; else the first mapped group (sorted)
    held = cfg.held_family or next((pmap[g] for g in sorted(rbps) if pmap.get(g, "?") != "?"), "?")
    test = tuple(g for g in rbps if pmap.get(g, "?") == held)
    train = tuple(g for g in rbps if g not in test)
    return RbpSplit(train=train, test=test, axis="paralog",
                    note=f"leave-out paralog group={held!r} (n_test={len(test)}); within-well transfer control")
