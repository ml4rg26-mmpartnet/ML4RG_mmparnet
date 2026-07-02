"""Independent-family scaling curve — the decisive interpolation test.

Train on N independent (mmseqs-clustered) RBP families, evaluate on HELD-OUT families, sweep N. This
operationalizes the reframe: paralog/window scaling is FLAT (assay-ceiling), but zero-shot transfer
should RISE with the number of INDEPENDENT families seen (~4 wells cannot interpolate; ~99-1744 can).

Heavy training runs on a GPU node (our M2 harness / the CORAL fork); this module is the model-side
INTERFACE + a loader that assembles the curve from per-N result files, so a notebook plots it from a
clone. Produce per-N `val_metrics.csv` (or a summary JSON), then `load_curve({N: path})`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def best_metric_from_val_csv(path, metric_col=1, header_skip=("Accuracy",)):
    """Best (max) value in `metric_col` of a per-epoch val_metrics.csv (F1 by default in the CORAL
    fork's schema). Skips header rows starting with any of `header_skip`."""
    best = float("nan")
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0] in header_skip:
                continue
            try:
                v = float(row[metric_col])
            except (ValueError, IndexError):
                continue
            best = v if (best != best or v > best) else best
    return best


def load_curve(points):
    """points: {N: path}. path -> .csv (best of metric_col) or .json ({'best_f1'|'f1'}).
    Returns sorted [(N, best_metric), ...]."""
    out = []
    for n, p in points.items():
        p = str(p)
        if p.endswith(".json"):
            d = json.loads(Path(p).read_text())
            v = d.get("best_f1", d.get("f1", float("nan")))
        else:
            v = best_metric_from_val_csv(p)
        out.append((int(n), float(v)))
    return sorted(out)


def slope(curve):
    """End-to-end slope (metric gain per log10 family-count). > 0 => independent-family diversity helps."""
    import math
    if len(curve) < 2:
        return float("nan")
    (n0, v0), (n1, v1) = curve[0], curve[-1]
    dl = math.log10(max(n1, 1)) - math.log10(max(n0, 1))
    return (v1 - v0) / dl if dl else float("nan")


def family_scaling(*_args, **_kwargs):
    raise NotImplementedError(
        "Training runs on a GPU node (M2 harness / CORAL fork). Produce per-N val_metrics files, then "
        "assemble with load_curve({10: 'famscale_10/fold_0/val_metrics.csv', 25: ..., 200: ...}).")
