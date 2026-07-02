"""Merge-health tests for the consolidated modular layer (the new functionality folded onto the base).

Fast + dependency-light: exercises the plug-in seam (registry), the leakage-controlled eval, the
CORAL/affinity metrics, the family split + clustering, and the CORAL adapter round-trip. No GPU, no
external data. Run: `pytest src/mmpartnet/tests/test_modular_merge.py`.
"""
from __future__ import annotations

import numpy as np


def test_config_new_keys_and_honesty_gate():
    from mmpartnet import config
    rc = config.RunConfig()
    assert rc.conditioning == "film" and rc.control == ()
    # default all-223 checkpoint is NOT a leave-out weight -> proxy-level, not a headline claim
    assert config.honest_zero_shot() is False


def test_registry_builds_every_head():
    from mmpartnet.models import list_heads, build_head, head_spec
    heads = list_heads()
    for name in ("conditioned", "early", "film", "perres", "xattn", "xattn2"):
        assert name in heads
    for name in heads:
        cls = build_head(name)               # lazy import must succeed for every registered head
        assert isinstance(cls, type)
    assert head_spec("xattn").owner == "dgu" and head_spec("xattn2").owner == "dfra"


def test_eval_metrics_and_controls():
    from mmpartnet.eval import roc_auc, profile_pearson, control_fired, family_disjoint_assert, shuffle_indices
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


def test_coral_and_affinity_metrics():
    from mmpartnet.metrics import coral_f1_auroc, validate_grid
    pred = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05]); y = np.array([1, 1, 1, 0, 0, 0])
    r = coral_f1_auroc(pred, y, seen_mask=np.array([1, 1, 0, 1, 0, 0], bool), best_thr=True)
    assert r["overall"]["auroc"] == 1.0 and r["unseen"]["n"] == 3
    vg = validate_grid(pred, np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1]), n_perm=200, seed=0)
    assert vg["effect"] > 0 and vg["p"] < 0.05


def test_splits_paralog_and_cluster():
    from mmpartnet.splits.registry import list_splits, get_split
    from mmpartnet.splits.base import SplitConfig
    from mmpartnet.cluster import n_families
    assert "paralog" in list_splits()
    sp = get_split(["A", "B", "C", "D"], SplitConfig(axis="paralog"),
                   meta={"paralog": {"A": "g1", "B": "g1", "C": "g2", "D": "g2"}})
    assert set(sp.test) & set(sp.train) == set()               # disjoint
    assert n_families({"A": "r1", "B": "r1", "C": "r2"}) == 2


def test_coral_adapter_roundtrip(tmp_path):
    from mmpartnet.adapters.coral import write_coral_csv, validate_roundtrip
    p = tmp_path / "coral.csv"
    write_coral_csv([("r1", "p1", 1, "ACGU", "MKV"), ("r1", "p2", 0, "ACGU", "MDE")], p)
    v = validate_roundtrip(p)
    assert v["pos"] == 1 and v["neg"] == 1 and v["n_prot"] == 2


def test_scaling_curve_loader(tmp_path):
    from mmpartnet.scaling import load_curve, best_metric_from_val_csv
    csvp = tmp_path / "val.csv"
    csvp.write_text("Accuracy,F1\n0.5,0.40\n0.6,0.55\n0.6,0.52\n")
    assert best_metric_from_val_csv(csvp) == 0.55
    assert load_curve({10: str(csvp), 50: str(csvp)}) == [(10, 0.55), (50, 0.55)]
