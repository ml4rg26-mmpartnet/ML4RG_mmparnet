"""Split strategies. Importing runs each module's `@register`.

  naive        POPULATED  no RBP holdout (train==test); pair with base.holdout_chrom for window split
  family       POPULATED  leave-one-family-out (ATtRACT families via io.cohort)
  rbp_holdout  STUB-ish   clean zero-shot: hold out cfg.held_rbps (the decisive M2 axis)
"""
from __future__ import annotations
from . import naive        # noqa: F401
from . import family       # noqa: F401
from . import rbp_holdout  # noqa: F401
