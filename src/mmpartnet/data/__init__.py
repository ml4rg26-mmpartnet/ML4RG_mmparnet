"""mmpartnet.data: modular, swappable data layer (sketch).

    from mmpartnet.data import DataConfig, get_source, iter_records
    src = get_source("encode_bigwig", DataConfig(group="AQR", cell="HepG2", nwin=20, target="density"))
    for rec in iter_records(src):
        rec["sequence"], rec["target"]   # feed your model

Switch backend by name: `get_source("encode_bam_counts", cfg)` (established crosslink counts, stub) /
`"hfds"` / `"local_pt"`. Switch target format via cfg.target (density | counts | hfds). See
`docs/DATA_INVENTORY.md` for what each source is a surrogate / established version of.
"""
from __future__ import annotations
from .base import DataConfig, DataSource, Window
from .registry import get_source, list_sources, register
from .loader import iter_records
from . import preprocess

# feature/rbp-binding-dataset: the (RNA-window, RBP) binary-binding dataset. Optional at import time —
# it pulls h5py/pandas and defaults to the lab data mount, so keep the base data package importable
# without those. Call build_dataset() only where the deps + data are present.
try:
    from .rbp_binding import RBPBindingDataset, build_dataset
except Exception:  # noqa: BLE001 — optional deps (h5py/pandas) or lab data mount absent
    RBPBindingDataset = build_dataset = None

__all__ = ["DataConfig", "DataSource", "Window", "get_source", "list_sources", "register",
           "iter_records", "preprocess", "RBPBindingDataset", "build_dataset"]
