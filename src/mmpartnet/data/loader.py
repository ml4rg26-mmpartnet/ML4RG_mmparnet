"""Turn a DataSource into uniform per-window records: {rbp, window, sequence, target}. The experiment then
encodes the sequence and runs the model; it does not care which source produced the data. Target
preprocessing (preprocess.to_profile) is applied here, so a source only returns RAW observed signal.
"""
from __future__ import annotations
import numpy as np

from .base import DataSource
from . import preprocess


def iter_records(source: DataSource):
    """Yield {rbp, window, sequence, target} for windows that pass the cfg.min_sum read filter.
    `target` is the preprocessed probability profile (kind = source.cfg.target). At most cfg.nwin
    records per RBP are emitted (matching the demo's 'score up to nwin windows'); a source may
    over-sample candidate windows so the read filter does not starve the count."""
    kind = source.cfg.target
    min_sum = source.cfg.min_sum
    nwin = source.cfg.nwin
    for rbp in source.rbps():
        kept = 0
        for w in source.windows(rbp):
            if kept >= nwin:
                break
            seq = source.sequence(w)
            if seq is None or len(seq) != source.cfg.lwin:
                continue
            obs = source.observed(rbp, w)
            if obs is None or float(np.abs(np.nan_to_num(obs)).sum()) < min_sum:
                continue
            kept += 1
            yield {"rbp": rbp, "window": w, "sequence": seq, "target": preprocess.to_profile(obs, kind)}
