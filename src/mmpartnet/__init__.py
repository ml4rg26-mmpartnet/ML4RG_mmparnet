"""mmpartnet — modular pipeline for ML4RG MultiModal PARNET.

The lab's PARNET (RNA-seq -> per-nt eCLIP profile) is loaded FROZEN; this repo validates the data +
recovery + interpretability pipeline around it. One internal data contract (the lab's
``ParnetDataElement``) is shared across the pipeline; the protein-conditioning direction ships as
swappable contracts (``mmpartnet.protein`` / ``splits`` / ``m2``) for the team to build on.

Layout (module -> role):
  config             central paths + RunConfig + the 4 gated swap-ins
  contract           ParnetDataElement (the single cross-team interface)
  data / protein / splits / m2   swappable layers (data source / protein rep / split axis / M2 scaffold)
  io.{genome,cohort,embeddings}  hg38 / ATtRACT families / ProteinRep
  adapters.{peaks,eclip_signal,eclip_counts}   ENCODE substrate -> per-nt target
  process.onehot     (4, L) one-hot encoding
  models.{parnet,ribex}   frozen PARNET base + protein representation
  experiments/*      the runnable bodies behind the demo notebooks

See README.md (overview) and CONTRACT.md (the data type + swap-in points).
"""
from __future__ import annotations

from . import config
from .config import RunConfig

__version__ = "0.1.0"
__all__ = ["config", "RunConfig", "__version__"]
