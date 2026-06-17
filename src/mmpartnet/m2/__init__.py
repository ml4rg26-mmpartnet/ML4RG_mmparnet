"""mmpartnet.m2: Milestone-2 scaffold (zero-shot cross-RBP generalization). Mostly stubs + contracts.

  leaveout_parnet.load_leaveout_parnet(held_rbps)  swap-in #1 seam + provenance guard (lab-gated)
  generalization.zero_shot_eval(...)               held-RBP eval + the two mandatory controls (stub)

M2 = an M1 FiLM-conditioned head trained under splits.rbp_holdout on a leave-out PARNET, evaluated
zero-shot with protein-shuffle + RNA-matched-N2 controls. The contract is fixed; bodies fill on M2 data.
"""
from __future__ import annotations
from . import leaveout_parnet, generalization

__all__ = ["leaveout_parnet", "generalization"]
