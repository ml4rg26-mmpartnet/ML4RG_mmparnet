"""Merge-health tests for the modular conditioning-head layer (registry seam + leakage-controlled eval).

Fast + dependency-light: exercises the plug-in seam (registry), the config honesty gate, the eval
contract (metrics + controls + family-disjoint guard), and the paralog split. No GPU, no external data.
The head-build check is torch-guarded so a torch-less runner still validates the rest.
Run: `pytest src/mmpartnet/tests/test_modular_merge.py`.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_config_new_keys_and_honesty_gate():
    from mmpartnet import config
    rc = config.RunConfig()
    assert rc.conditioning == "film" and rc.control == ()
    # default all-223 checkpoint is NOT a leave-out weight -> proxy-level, not a headline claim
    assert config.honest_zero_shot() is False


def test_registry_lists_team_heads():
    from mmpartnet.models import list_heads, head_spec
    heads = list_heads()
    for name in ("early", "film", "xattn", "xattn2"):
        assert name in heads
    # our unpublished research heads must NOT be in the team base
    for name in ("conditioned", "perres", "perres_bidir"):
        assert name not in heads
    assert head_spec("xattn").owner == "dgu" and head_spec("xattn2").owner == "dfra"
    assert head_spec("early").owner == "cgerards"


def test_registry_builds_every_head():
    pytest.importorskip("torch")
    from mmpartnet.models import list_heads, build_head
    for name in list_heads():
        cls = build_head(name)               # lazy import must succeed for every registered head
        assert isinstance(cls, type)


def test_eval_metrics_and_controls():
    from mmpartnet.eval import (roc_auc, profile_pearson, control_fired,
                                family_disjoint_assert, shuffle_indices)
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert profile_pearson([1, 2, 3, 4], [1, 2, 3, 4]) > 0.99
    assert control_fired(0.80, 0.51)["fired"] is True          # shuffle collapses -> control fired
    assert control_fired(0.80, 0.79)["warn"] is True           # shuffle did nothing -> flagged
    assert all(shuffle_indices(8, 0) != np.arange(8))          # derangement: no fixed point
    # family leakage must be caught
    try:
        family_disjoint_assert(["A", "B"], ["C", "A"], {"A": "f1", "B": "f2", "C": "f1"})
        raise AssertionError("family_disjoint_assert failed to catch overlap")
    except AssertionError as e:
        assert "family leakage" in str(e)


def test_splits_paralog():
    from mmpartnet.splits.registry import list_splits, get_split
    from mmpartnet.splits.base import SplitConfig
    assert "paralog" in list_splits()
    sp = get_split(["A", "B", "C", "D"], SplitConfig(axis="paralog"),
                   meta={"paralog": {"A": "g1", "B": "g1", "C": "g2", "D": "g2"}})
    assert set(sp.test) & set(sp.train) == set()               # disjoint
    assert len(sp.test) >= 1 and len(sp.train) >= 1
