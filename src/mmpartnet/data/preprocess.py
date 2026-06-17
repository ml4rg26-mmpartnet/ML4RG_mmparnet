"""Per-format target preprocessing: raw observed per-nt signal -> a probability profile over the window's
positions, which is what the model predicts (softmax over the window) and what Pearson/Spearman score.
Add a format by registering a function; the loader calls `to_profile(obs, kind)`.
"""
from __future__ import annotations
import numpy as np

_PRE: dict = {}


def register(kind: str):
    def deco(fn):
        _PRE[kind] = fn
        return fn
    return deco


@register("density")
def _density(obs):
    """ENCODE 'signal of unique reads' read-density (RPM) or any non-negative track: abs, normalize to a
    position-distribution. The scale (RPM vs raw) cancels under this normalization (Pearson is scale-inv.)."""
    v = np.abs(np.nan_to_num(np.asarray(obs, float)))
    s = v.sum()
    return v / s if s > 0 else v


@register("counts")
def _counts(obs):
    """Single-nt 5'-read-start crosslink COUNTS (the established target; built from the eCLIP BAMs). Same
    normalization to a profile; sharper than density (no read-footprint blur)."""
    v = np.nan_to_num(np.asarray(obs, float))
    v[v < 0] = 0.0
    s = v.sum()
    return v / s if s > 0 else v


@register("hfds")
def _hfds(obs):
    raise NotImplementedError("preprocess 'hfds': fill when the lab encode.filtered/HFDS source lands")


def to_profile(obs, kind: str):
    if kind not in _PRE:
        raise KeyError(f"unknown target kind {kind!r}; available: {sorted(_PRE)}")
    return _PRE[kind](obs)


def list_kinds() -> list:
    return sorted(_PRE)
