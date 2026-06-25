"""Protein+cell FiLM head on top of frozen PARNET RNA features.

The first multimodal model keeps the pretrained PARNET RNA body frozen and learns
a small profile head conditioned on both the RBP protein representation and the
cell line.  It predicts RBPNet-style target/control/total per-position profiles.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ProteinCellFiLMProfileHead(nn.Module):
    """FiLM-condition PARNET body features with protein and cell context.

    Inputs:
      rna_features: ``[B, C, L]`` from ``ParnetModel.body_feats``
      protein_embedding: ``[B, P]`` pooled protein representation
      cell_index: ``[B]`` integer cell-line ids

    Returns log-probability tracks plus probabilities for evaluation:
      ``target``, ``control``, ``total``: ``[B, L]`` probability profiles
      ``*_logprob``: matching log-profiles
      ``mix_coeff``: ``[B]`` target/control mixture coefficient
    """

    def __init__(
        self,
        protein_dim: int,
        *,
        rna_channels: int = 512,
        cell_count: int = 2,
        cell_dim: int = 32,
        hidden_dim: int = 256,
        mix_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rna_channels = rna_channels
        self.cell_embedding = nn.Embedding(cell_count, cell_dim)
        self.conditioner = nn.Sequential(
            nn.LayerNorm(protein_dim + cell_dim),
            nn.Linear(protein_dim + cell_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * rna_channels),
        )
        self.target = nn.Conv1d(rna_channels, 1, kernel_size=1)
        self.control = nn.Conv1d(rna_channels, 1, kernel_size=1)
        self.mix = nn.Sequential(
            nn.Linear(rna_channels, mix_hidden_dim),
            nn.ReLU(),
            nn.Linear(mix_hidden_dim, 1),
        )

    def _condition(self, rna_features: torch.Tensor, protein_embedding: torch.Tensor, cell_index: torch.Tensor):
        cell = self.cell_embedding(cell_index)
        cond = torch.cat([protein_embedding, cell], dim=1)
        gamma_beta = self.conditioner(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = 1.0 + torch.tanh(gamma).unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * rna_features + beta, cond

    def forward(
        self,
        rna_features: torch.Tensor,
        protein_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused, cond = self._condition(rna_features, protein_embedding, cell_index)
        target_logits = self.target(fused).squeeze(1)
        control_logits = self.control(fused).squeeze(1)
        if mask is not None:
            target_logits = target_logits.masked_fill(~mask, torch.finfo(target_logits.dtype).min)
            control_logits = control_logits.masked_fill(~mask, torch.finfo(control_logits.dtype).min)

        target_logprob = torch.log_softmax(target_logits, dim=-1)
        control_logprob = torch.log_softmax(control_logits, dim=-1)
        if mask is None:
            mix_input = fused.mean(dim=-1)
        else:
            mask_f = mask.to(dtype=fused.dtype).unsqueeze(1)
            mix_input = (fused * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
        mix_coeff = torch.sigmoid(self.mix(mix_input)).squeeze(-1)
        mix = mix_coeff.unsqueeze(-1)

        max_logprob = torch.maximum(target_logprob, control_logprob)
        total_logprob = max_logprob + torch.log(
            mix * torch.exp(target_logprob - max_logprob)
            + (1.0 - mix) * torch.exp(control_logprob - max_logprob)
            + 1e-10
        )

        return {
            "target_logprob": target_logprob,
            "control_logprob": control_logprob,
            "total_logprob": total_logprob,
            "target": torch.exp(target_logprob),
            "control": torch.exp(control_logprob),
            "total": torch.exp(total_logprob),
            "mix_coeff": mix_coeff,
        }

    def loss(
        self,
        rna_features: torch.Tensor,
        protein_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        eclip_counts: torch.Tensor,
        control_counts: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        min_count: float = 10.0,
        mix_penalty: float = 0.0,
    ) -> torch.Tensor:
        out = self.forward(rna_features, protein_embedding, cell_index, mask=mask)
        eclip_depth = eclip_counts.sum(dim=-1)
        control_depth = control_counts.sum(dim=-1)
        eclip_keep = eclip_depth >= min_count
        control_keep = control_depth >= min_count

        eclip_nll = -(eclip_counts * out["total_logprob"]).sum(dim=-1) / eclip_depth.clamp_min(1.0)
        control_nll = -(control_counts * out["control_logprob"]).sum(dim=-1) / control_depth.clamp_min(1.0)
        loss = (eclip_nll * eclip_keep).sum() / eclip_keep.sum().clamp_min(1)
        loss = loss + (control_nll * control_keep).sum() / control_keep.sum().clamp_min(1)
        if mix_penalty:
            loss = loss + mix_penalty * out["mix_coeff"].mean()
        return loss


__all__ = ["ProteinCellFiLMProfileHead"]
