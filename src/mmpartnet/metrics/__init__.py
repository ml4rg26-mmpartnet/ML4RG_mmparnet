"""Comparison metrics against external baselines (CORAL) + the affinity certificate."""
from __future__ import annotations

from .coral import coral_f1_auroc, validate_grid

__all__ = ["coral_f1_auroc", "validate_grid"]
