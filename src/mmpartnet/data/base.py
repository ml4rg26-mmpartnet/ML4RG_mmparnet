"""Modular data layer for mmpartnet (team sketch, fill as needed).

Goal: swap data SOURCES (ENCODE bigWig now; BAM crosslink-counts / lab HFDS / encode.filtered later) behind
one interface, and keep FORMAT preprocessing in one place, so experiments do not care where the per-nt
profile came from. Populate only the parts a given experiment needs; leave the rest as stubs.

Flow: a `DataSource` yields, per RBP, a set of `Window`s; for each window it returns the input sequence and
the raw observed per-nt target. `loader.iter_records` applies the target preprocessing (`preprocess.py`) and
hands experiments a uniform record. Register a source in `registry.py` so `--source <name>` switches it.

This is a SKETCH: interfaces are meant to be stable, bodies are meant to be filled in by whoever needs a
given source/format. The `encode_bigwig` source is populated for the current demo.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class Window:
    chrom: str
    start: int
    end: int
    strand: str = "+"
    rbp: Optional[str] = None


@dataclass
class DataConfig:
    """One experiment's data spec. `source` selects the DataSource (registry name); `target` selects the
    preprocessing kind (preprocess.py)."""
    source: str = "encode_bigwig"
    group: str = "spliceosome"      # io.groups name (or a comma-list of RBP symbols)
    cell: str = "HepG2"
    lwin: int = 600                 # window length (PARNET tiles are 600 nt)
    nwin: int = 30                  # windows per RBP
    target: str = "density"         # preprocess kind: density | counts | hfds (see preprocess.py)
    min_sum: float = 10.0           # drop windows with observed sum below this (demo >=10-read filter)
    extra: dict = field(default_factory=dict)


class DataSource(ABC):
    """Swappable source of (window -> sequence + observed per-nt target). Implement the four methods;
    everything format-specific (counts vs density vs hfds) lives behind `cfg.target` + `preprocess.py`."""

    #: the natural target format this source emits; the loader can override via cfg.target
    target_kind: str = "density"

    def __init__(self, cfg: "DataConfig | None" = None):
        self.cfg = cfg or DataConfig(source=getattr(self, "name", "encode_bigwig"))

    @abstractmethod
    def rbps(self) -> list:
        """RBP symbols available under this source + the cfg group/cell."""

    @abstractmethod
    def windows(self, rbp: str) -> Iterable[Window]:
        """Up to cfg.nwin windows for `rbp`."""

    @abstractmethod
    def sequence(self, w: Window) -> Optional[str]:
        """Input sequence for a window (transcript 5'->3'); None if unavailable."""

    @abstractmethod
    def observed(self, rbp: str, w: Window):
        """Raw observed per-nt target over the window (np.ndarray, len cfg.lwin); None if unavailable.
        Preprocessing to a probability profile is applied by loader.iter_records via preprocess.py."""
