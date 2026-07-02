"""Early-fusion M1 baseline (the supervisor's literal ask) — the floor the
FiLM-conditioned head must beat.

The simplest possible multimodal model: concatenate the pooled RNA representation
(frozen-PARNET body features, 512-d) with the protein representation (ESM2-650
pooled, 640-d) and feed a 2-layer MLP -> a single bound/not-bound logit (BCE).

The DIAGNOSTIC purpose is as important as the baseline: concat lets the model route
around the protein via the RNA features, so under the protein-permutation gate the
anti-bypass gap is expected to be ~0 on N2 — empirically motivating the N2 contrast +
FiLM conditioning in `models.heads.ConditionedHead`. Always report this model BESIDE
the two standing controls (protein-shuffle + RNA-matched hard-N2).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    """concat(RNA-pooled, protein) -> 2-layer MLP -> 1 logit (bound/not-bound)."""

    def __init__(self, dr: int = 512, dp: int = 640, h: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dr + dp, h),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(h, 1),
        )

    def forward(self, gr: torch.Tensor, ep: torch.Tensor) -> torch.Tensor:
        """gr: (B, dr) pooled RNA feats; ep: (B, dp) protein rep. Returns (B,) logits."""
        return self.net(torch.cat([gr, ep], dim=-1)).squeeze(-1)


def fit_predict(model, gr, ep, y, *, epochs: int = 8, lr: float = 1e-3, wd: float = 1e-4,
                device: str = "cpu"):
    """Minimal BCE training loop for the baseline. gr/ep/y are tensors on the same split.
    Returns the per-example predicted probabilities (detached numpy)."""
    model = model.to(device)
    gr, ep, y = gr.to(device), ep.to(device), y.float().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(model(gr, ep), y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(gr, ep)).cpu().numpy()
