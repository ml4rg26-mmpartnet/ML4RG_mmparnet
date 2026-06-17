"""Source registry: name -> DataSource class, so `--source <name>` / `get_source(name)` switches the data
backend with no experiment changes. Sources self-register via the `@register` decorator (see data/sources)."""
from __future__ import annotations
from .base import DataConfig, DataSource

_SOURCES: dict = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _SOURCES[name] = cls
        return cls
    return deco


def _ensure_loaded():
    if not _SOURCES:
        from . import sources  # noqa: F401  (import triggers @register on each source module)


def get_source(name: str, cfg: "DataConfig | None" = None) -> DataSource:
    _ensure_loaded()
    if name not in _SOURCES:
        raise KeyError(f"unknown data source {name!r}; available: {sorted(_SOURCES)}")
    return _SOURCES[name](cfg or DataConfig(source=name))


def list_sources() -> list:
    _ensure_loaded()
    return sorted(_SOURCES)
