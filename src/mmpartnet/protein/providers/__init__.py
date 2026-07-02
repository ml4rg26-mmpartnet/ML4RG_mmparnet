"""Protein-rep providers. Importing runs each module's `@register`.

  esm650_pooled  POPULATED  ESM2-650 pooled (640-d), conservative baseline rep
  ribex_proxy    POPULATED  ESM2-650 (+) STRING-PE (704-d), the proxy used for the initial real run
  prott5_h5      POPULATED  pooled ProtT5 H5 vectors used by the FiLM baseline
  ribex_real     STUB       lab-trained RIBEX fused embedding (drop-in via env ML4RG_RIBEX)
"""
from __future__ import annotations
from . import esm_pooled    # noqa: F401
from . import ribex_proxy   # noqa: F401
from . import prott5_h5     # noqa: F401
from . import ribex_real    # noqa: F401
