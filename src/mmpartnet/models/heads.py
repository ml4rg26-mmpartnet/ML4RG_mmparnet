"""Milestone heads. The TFBindFormer-equivalent for our scope = condition the frozen PARNET
RNA representation on the protein rep and train to discriminate on the N2 (protein-specific)
axis. Per Moyon (less architecture, more loss/signal) the fusion is a simple FiLM; the
anti-bypass property comes from the N2 contrast + the protein-permutation gate, not the block.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConditionedHead(nn.Module):
    """score(g_r, e_p): RNA window feature (frozen PARNET body, pooled) FiLM-modulated by the
    protein rep. Optional RNA-only residual branch (the bias-mixture analog)."""

    def __init__(self, dr=512, dp=640, h=128, residual=True):
        super().__init__()
        self.residual = residual
        self.rna = nn.Linear(dr, h)
        self.film = nn.Linear(dp, 2 * h)
        self.delta = nn.Sequential(nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.rna_only = nn.Sequential(nn.Linear(dr, h), nn.ReLU(), nn.Linear(h, 1)) if residual else None

    def forward(self, gr, ep):
        gamma, beta = self.film(ep).chunk(2, -1)
        h = gamma * self.rna(gr) + beta                           # fused (protein-conditioned) rep
        d = self.delta(h).squeeze(-1)                             # protein-conditioned increment
        if self.residual:
            b = self.rna_only(gr).squeeze(-1)                     # RNA-only bindability
            return b.detach() + d, b, h
        return d, None, h
