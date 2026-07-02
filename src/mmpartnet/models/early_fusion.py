"""Concat-fusion baseline head on top of frozen PARNET RNA features.

The simplest possible sibling to ProteinCellFiLMProfileHead:
  - pool the PARNET body features over RNA positions  -> (B, 512)
  - concat with the protein embedding                 -> (B, 512 + protein_dim)
  - concat with a learned cell-line embedding         -> (B, 512 + protein_dim + cell_dim)
  - feed through a small MLP                          -> (B, 1) binding logit

There is no FiLM gating, no per-position conditioning, no profile head — that is
deliberate. This is the early-fusion baseline that the FiLM / cross-attention
variants have to beat. Same forward signature as ProteinCellFiLMProfileHead so it
plugs into experiments/film_multitask.py via an ``--arch`` switch with no
training-loop changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TASKS = {"multitask", "binary-only", "profile-only"}


class EarlyFusionConcatHead(nn.Module):
    """concat[pooled_RNA, protein_embedding, cell_embedding] -> MLP -> binding logit.

    Only the ``binary-only`` task is supported. The other task names are accepted
    for interface compatibility with ProteinCellFiLMProfileHead but no profile head
    is computed — the early-fusion design pools RNA positions away, which discards
    the per-position information a profile head would need.

    Args:
      protein_dim:    width of the per-RBP protein vector (e.g. 1024 for ProtT5)
      rna_channels:   width of the PARNET body output (default 512)
      cell_count:     number of distinct cell lines to embed (HepG2 + K562 = 2)
      cell_dim:       width of the learned cell-line embedding
      hidden:         hidden width of the MLP
      dropout:        dropout probability between MLP layers
    """

    def __init__(
        self,
        protein_dim: int,
        *,
        rna_channels: int = 512,
        cell_count: int = 2,
        cell_dim: int = 32,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rna_channels = rna_channels
        self.cell_embedding = nn.Embedding(cell_count, cell_dim)

        # concat dim = pooled RNA (rna_channels) + protein vec + cell vec
        in_dim = rna_channels + protein_dim + cell_dim

        # 2-hidden-layer MLP. GELU is the modern default (smoother than ReLU);
        # the FiLM head uses GELU too, so we match for fair comparison.
        self.binding = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        rna_features: torch.Tensor,
        protein_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        task: str = "binary-only",
    ) -> dict[str, torch.Tensor]:
        """Run the head on one batch.

        Args:
          rna_features:      (B, C, L) from ParnetModel.body_feats — C = 512, L = 600
          protein_embedding: (B, P)    pooled protein vector for this batch's RBPs
          cell_index:        (B,)      integer cell-line ids (0=HepG2, 1=K562)
          mask:              (B, L)    True at real positions, False at padding
          task:              ignored — only here so the training loop can pass it

        Returns:
          dict with ``binding_logit`` and ``binding_prob`` (each shape (B,))
        """
        pooled = self._masked_mean(rna_features, mask)        # (B, C)
        cell = self.cell_embedding(cell_index)                # (B, cell_dim)
        fused = torch.cat([pooled, protein_embedding, cell], dim=-1)  # (B, C + P + cell_dim)

        # compute once, reuse for both keys
        logit = self.binding(fused).squeeze(-1)               # (B,)
        return {
            "binding_logit": logit,
            "binding_prob": torch.sigmoid(logit),
        }

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Mean-pool over the last axis, ignoring padded positions.

        features: (B, C, L), mask: (B, L) bool. Returns (B, C).
        """
        if mask is None:
            return features.mean(dim=-1)
        mask_f = mask.to(dtype=features.dtype).unsqueeze(1)   # (B, 1, L)
        return (features * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)

    def loss_components(
        self,
        rna_features: torch.Tensor,
        protein_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        eclip_counts: torch.Tensor,
        control_counts: torch.Tensor,
        binding_label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        binary_pos_weight: float | None = None,
        task: str = "binary-only",
        **_,                              # swallow FiLM-only kwargs (lambda_profile etc.)
    ) -> dict[str, torch.Tensor]:
        """Forward + BCE loss. Returns the same dict shape as the FiLM head's
        ``loss_components`` so the training loop in experiments/film_multitask.py
        can consume it identically.

        Profile-related fields (``profile_loss``, ``profile_n``) are returned as
        zeros because this head has no profile output. The training-loop metric
        code already handles ``profile_n == 0``.
        """
        out = self.forward(rna_features, protein_embedding, cell_index, mask=mask, task=task)
        pos_weight = None if binary_pos_weight is None else eclip_counts.new_tensor(binary_pos_weight)
        binary_loss = F.binary_cross_entropy_with_logits(
            out["binding_logit"], binding_label.float(), pos_weight=pos_weight,
        )
        return {
            "loss": binary_loss,
            "profile_loss": eclip_counts.new_tensor(0.0),
            "binary_loss": binary_loss,
            "profile_n": eclip_counts.new_tensor(0, dtype=torch.long),
        }

    def loss(self, *args, **kwargs) -> torch.Tensor:
        """Thin convenience wrapper: returns just the scalar ``loss`` tensor.

        Matches ProteinCellFiLMProfileHead.loss so a caller can use either head
        interchangeably.
        """
        return self.loss_components(*args, **kwargs)["loss"]


__all__ = ["EarlyFusionConcatHead"]
