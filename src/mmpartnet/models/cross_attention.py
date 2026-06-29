"""Cell-FiLM bidirectional RNA/protein cross-attention head.

This head keeps the pretrained PARNET RNA body frozen and learns a small
TFBindFormer-like fusion module. Each block first conditions RNA tokens on cell
line with FiLM, then updates RNA and protein tokens with synchronous
bidirectional cross-attention. A final RNA-to-protein attention pass produces an
RNA-centric representation for RBPNet-style profile prediction and binding
classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttentionUpdate(nn.Module):
    """Cross-attention + residual/norm + FFN update for one query stream."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attn(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        out = self.attn_norm(query + self.dropout(attended))
        out = self.ffn_norm(out + self.dropout(self.ffn(out)))
        return out, weights if need_weights else None


class _CellFiLMBidirectionalBlock(nn.Module):
    """Cell-conditioned RNA plus synchronous RNA/protein cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        cell_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.cell_to_film = nn.Linear(cell_dim, 2 * hidden_dim)
        self.rna_cell_norm = nn.LayerNorm(hidden_dim)
        self.protein_attends_rna = _CrossAttentionUpdate(hidden_dim, num_heads, ff_dim, dropout)
        self.rna_attends_protein = _CrossAttentionUpdate(hidden_dim, num_heads, ff_dim, dropout)

    def forward(
        self,
        rna: torch.Tensor,
        protein: torch.Tensor,
        cell: torch.Tensor,
        *,
        rna_mask: torch.Tensor | None = None,
        protein_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.cell_to_film(cell).chunk(2, dim=-1)
        scale = 1.0 + torch.tanh(gamma).unsqueeze(1)
        rna_cell = self.rna_cell_norm(scale * rna + beta.unsqueeze(1))

        rna_key_padding_mask = None if rna_mask is None else ~rna_mask.bool()
        protein_key_padding_mask = None if protein_mask is None else ~protein_mask.bool()
        protein_next, _ = self.protein_attends_rna(
            protein,
            rna_cell,
            key_padding_mask=rna_key_padding_mask,
        )
        rna_next, _ = self.rna_attends_protein(
            rna_cell,
            protein,
            key_padding_mask=protein_key_padding_mask,
        )
        return rna_next, protein_next


class _LatentProteinCompressor(nn.Module):
    """Compress variable-length protein tokens with learned latent queries."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        latent_len: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        if latent_len <= 0:
            raise ValueError("latent_len must be positive")
        self.latent_len = latent_len
        self.latent_queries = nn.Parameter(torch.randn(latent_len, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        protein: torch.Tensor,
        protein_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = protein.shape[0]
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None if protein_mask is None else ~protein_mask.bool()
        attended, _ = self.attn(
            queries,
            protein,
            protein,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.attn_norm(queries + self.dropout(attended))
        out = self.ffn_norm(out + self.dropout(self.ffn(out)))
        out_mask = torch.ones(
            batch_size,
            self.latent_len,
            dtype=torch.bool,
            device=protein.device,
        )
        return out, out_mask


class ProteinCellCrossAttentionProfileHead(nn.Module):
    """Cell-FiLM + bidirectional cross-attend RNA/protein residue embeddings.

    Inputs:
      rna_features: ``[B, C, L]`` from ``ParnetModel.body_feats``
      protein_residue_embedding: ``[B, Lp, P]`` padded ProtT5 residues
      protein_mask: ``[B, Lp]`` valid protein residue positions
      cell_index: ``[B]`` integer cell-line ids
    """

    def __init__(
        self,
        protein_dim: int,
        *,
        rna_channels: int = 512,
        cell_count: int = 2,
        cell_dim: int = 32,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 1,
        ff_dim: int | None = None,
        mix_hidden_dim: int = 128,
        dropout: float = 0.1,
        protein_projection_hidden_dim: int | None = 768,
        protein_compression: str = "latent",
        protein_latent_len: int = 256,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if protein_compression not in {"none", "latent"}:
            raise ValueError("protein_compression must be 'none' or 'latent'")
        self.rna_channels = rna_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.protein_compression = protein_compression
        self.protein_latent_len = protein_latent_len
        self.cell_embedding = nn.Embedding(cell_count, cell_dim)
        if rna_channels == hidden_dim:
            self.rna_projection = nn.Identity()
        else:
            self.rna_projection = nn.Linear(rna_channels, hidden_dim)
        if protein_projection_hidden_dim is None or protein_projection_hidden_dim <= 0:
            self.protein_projection = nn.Linear(protein_dim, hidden_dim)
        else:
            self.protein_projection = nn.Sequential(
                nn.Linear(protein_dim, protein_projection_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(protein_projection_hidden_dim, hidden_dim),
            )
        self.rna_norm = nn.LayerNorm(hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        ff_dim = ff_dim or hidden_dim * 4
        self.protein_compressor = (
            _LatentProteinCompressor(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                latent_len=protein_latent_len,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            if protein_compression == "latent"
            else None
        )
        self.blocks = nn.ModuleList(
            [
                _CellFiLMBidirectionalBlock(
                    hidden_dim=hidden_dim,
                    cell_dim=cell_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_rna_attends_protein = _CrossAttentionUpdate(hidden_dim, num_heads, ff_dim, dropout)
        self.target = nn.Linear(hidden_dim, 1)
        self.control = nn.Linear(hidden_dim, 1)
        self.mix = nn.Sequential(
            nn.Linear(hidden_dim, mix_hidden_dim),
            nn.ReLU(),
            nn.Linear(mix_hidden_dim, 1),
        )
        self.binary_position_score = nn.Linear(hidden_dim, 1)
        self.binary_gate = nn.Sequential(
            nn.Linear(hidden_dim, mix_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mix_hidden_dim, 1),
        )
        self.binding = nn.Sequential(
            nn.Linear(hidden_dim, mix_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mix_hidden_dim, 1),
        )

    def _fuse(
        self,
        rna_features: torch.Tensor,
        protein_residue_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        mask: torch.Tensor | None,
        protein_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        rna_tokens = rna_features.transpose(1, 2)
        valid_cell = cell_index >= 0
        cell = self.cell_embedding(cell_index.clamp_min(0))
        cell = cell * valid_cell.to(dtype=cell.dtype).unsqueeze(-1)
        rna = self.rna_norm(self.rna_projection(rna_tokens))
        protein = self.protein_norm(self.protein_projection(protein_residue_embedding))
        if self.protein_compressor is not None:
            protein, protein_mask = self.protein_compressor(protein, protein_mask)
        for block in self.blocks:
            rna, protein = block(
                rna,
                protein,
                cell,
                rna_mask=mask,
                protein_mask=protein_mask,
            )
        protein_key_padding_mask = None if protein_mask is None else ~protein_mask.bool()
        fused, _ = self.final_rna_attends_protein(
            rna,
            protein,
            key_padding_mask=protein_key_padding_mask,
        )
        return fused

    def forward(
        self,
        rna_features: torch.Tensor,
        protein_residue_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        mask: torch.Tensor | None = None,
        protein_mask: torch.Tensor | None = None,
        *,
        task: str = "multitask",
    ) -> dict[str, torch.Tensor]:
        if task not in {"multitask", "profile-only", "binary-only"}:
            raise ValueError("task must be one of: multitask, profile-only, binary-only")
        fused = self._fuse(rna_features, protein_residue_embedding, cell_index, mask, protein_mask)
        out: dict[str, torch.Tensor] = {}
        pooled = self._masked_mean(fused, mask)

        if task in {"multitask", "profile-only"}:
            target_logits = self.target(fused).squeeze(-1)
            control_logits = self.control(fused).squeeze(-1)
            if mask is not None:
                target_logits = target_logits.masked_fill(~mask, torch.finfo(target_logits.dtype).min)
                control_logits = control_logits.masked_fill(~mask, torch.finfo(control_logits.dtype).min)

            target_logprob = torch.log_softmax(target_logits, dim=-1)
            control_logprob = torch.log_softmax(control_logits, dim=-1)
            target_prob = torch.exp(target_logprob)
            mix_coeff = torch.sigmoid(self.mix(pooled)).squeeze(-1)
            mix = mix_coeff.unsqueeze(-1)
            max_logprob = torch.maximum(target_logprob, control_logprob)
            total_logprob = max_logprob + torch.log(
                mix * torch.exp(target_logprob - max_logprob)
                + (1.0 - mix) * torch.exp(control_logprob - max_logprob)
                + 1e-10
            )
            out.update(
                {
                    "target_logprob": target_logprob,
                    "control_logprob": control_logprob,
                    "total_logprob": total_logprob,
                    "target": target_prob,
                    "control": torch.exp(control_logprob),
                    "total": torch.exp(total_logprob),
                    "mix_coeff": mix_coeff,
                }
            )

        if task in {"multitask", "binary-only"}:
            binary_position_logits = self.binary_position_score(fused).squeeze(-1)
            if mask is not None:
                binary_position_logits = binary_position_logits.masked_fill(
                    ~mask,
                    torch.finfo(binary_position_logits.dtype).min,
                )
            binary_position_prob = torch.softmax(binary_position_logits, dim=-1)
            if task == "multitask":
                binding_gate = torch.sigmoid(self.binary_gate(pooled)).squeeze(-1)
                alpha_bind = binding_gate.unsqueeze(-1) * out["target"] + (
                    1.0 - binding_gate.unsqueeze(-1)
                ) * binary_position_prob
                out["binding_gate"] = binding_gate
            else:
                alpha_bind = binary_position_prob
            binding_input = (fused * alpha_bind.unsqueeze(-1)).sum(dim=1)
            binding_logit = self.binding(binding_input).squeeze(-1)
            out.update(
                {
                    "binary_position_prob": binary_position_prob,
                    "alpha_bind": alpha_bind,
                    "binding_logit": binding_logit,
                    "binding_prob": torch.sigmoid(binding_logit),
                }
            )
        return out

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        mask_f = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

    def loss_components(
        self,
        rna_features: torch.Tensor,
        protein_residue_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        eclip_counts: torch.Tensor,
        control_counts: torch.Tensor,
        binding_label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        protein_mask: torch.Tensor | None = None,
        min_count: float = 10.0,
        mix_penalty: float = 0.0,
        lambda_binary: float = 1.0,
        lambda_profile: float = 1.0,
        profile_mask: torch.Tensor | None = None,
        binary_pos_weight: float | None = None,
        task: str = "multitask",
    ) -> dict[str, torch.Tensor]:
        out = self.forward(
            rna_features,
            protein_residue_embedding,
            cell_index,
            mask=mask,
            protein_mask=protein_mask,
            task=task,
        )
        if mask is not None:
            count_mask = mask.to(dtype=eclip_counts.dtype)
            eclip_counts = eclip_counts * count_mask
            control_counts = control_counts * count_mask
        eclip_depth = eclip_counts.sum(dim=-1)
        control_depth = control_counts.sum(dim=-1)
        if profile_mask is None:
            profile_keep = eclip_depth >= min_count
        else:
            profile_keep = profile_mask.bool()
        if task == "binary-only":
            profile_keep = torch.zeros_like(profile_keep, dtype=torch.bool)

        profile_loss = eclip_counts.new_tensor(0.0)
        if task in {"multitask", "profile-only"}:
            eclip_nll = -(eclip_counts * out["total_logprob"]).sum(dim=-1) / eclip_depth.clamp_min(1.0)
            control_nll = -(control_counts * out["control_logprob"]).sum(dim=-1) / control_depth.clamp_min(1.0)
            profile_loss = (eclip_nll * profile_keep).sum() / profile_keep.sum().clamp_min(1)
            profile_loss = profile_loss + (control_nll * profile_keep).sum() / profile_keep.sum().clamp_min(1)
        binary_loss = eclip_counts.new_tensor(0.0)
        if task in {"multitask", "binary-only"} and binding_label is not None:
            pos_weight = None
            if binary_pos_weight is not None:
                pos_weight = eclip_counts.new_tensor(binary_pos_weight)
            binary_loss = F.binary_cross_entropy_with_logits(
                out["binding_logit"],
                binding_label.to(dtype=out["binding_logit"].dtype),
                pos_weight=pos_weight,
            )
        loss = lambda_profile * profile_loss + lambda_binary * binary_loss
        if mix_penalty and "mix_coeff" in out:
            loss = loss + mix_penalty * out["mix_coeff"].mean()
        return {
            "loss": loss,
            "profile_loss": profile_loss,
            "binary_loss": binary_loss,
            "profile_n": profile_keep.sum(),
        }


__all__ = ["ProteinCellCrossAttentionProfileHead"]
