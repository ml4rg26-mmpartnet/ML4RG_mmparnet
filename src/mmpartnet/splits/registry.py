"""Split-strategy registry: name -> fn(rbps, cfg, meta=None) -> RbpSplit (parallel to data.registry)."""
from __future__ import annotations
from .base import SplitConfig, RbpSplit

_STRATS: dict = {}


def register(name: str):
    def deco(fn):
        _STRATS[name] = fn
        return fn
    return deco


def _ensure_loaded():
    if not _STRATS:
        from . import strategies  # noqa: F401  (import triggers @register)


def get_split(rbps, cfg: "SplitConfig | None" = None, meta=None) -> RbpSplit:
    _ensure_loaded()
    cfg = cfg or SplitConfig()
    if cfg.axis not in _STRATS:
        raise KeyError(f"unknown split axis {cfg.axis!r}; available: {sorted(_STRATS)}")
    return _STRATS[cfg.axis](list(rbps), cfg, meta)


def list_splits() -> list:
    _ensure_loaded()
    return sorted(_STRATS)
