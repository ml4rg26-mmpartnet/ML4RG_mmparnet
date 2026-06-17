"""Protein-source registry: name -> ProteinSource (parallel to data.registry)."""
from __future__ import annotations
from .base import ProteinConfig, ProteinSource

_SOURCES: dict = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _SOURCES[name] = cls
        return cls
    return deco


def _ensure_loaded():
    if not _SOURCES:
        from . import providers  # noqa: F401  (import triggers @register)


def get_protein(name: str = "ribex_proxy", cfg: "ProteinConfig | None" = None) -> ProteinSource:
    _ensure_loaded()
    if name not in _SOURCES:
        raise KeyError(f"unknown protein rep {name!r}; available: {sorted(_SOURCES)}")
    return _SOURCES[name](cfg or ProteinConfig(mode=name))


def list_proteins() -> list:
    _ensure_loaded()
    return sorted(_SOURCES)
