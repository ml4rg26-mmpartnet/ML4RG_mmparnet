"""Cell-FiLM + TFBindFormer-style bidirectional RNA/protein cross-attention head.

Architecture (in order):
  1. Cell FiLM  — modulate RNA tokens once with cell-line context
  2. ProteinCompressor  — compress ProtT5 residues [B, Lp, 1024] to a fixed
     target_prot_len via learned queries (TFBindFormer: ProteinReduceVariable)
  3. HybridCrossAttentionEncoder  — num_layers total; the first num_bidir_layers
     update both RNA and protein, later layers update RNA only
     (TFBindFormer: asymmetric bidirectionality)
  4. Profile heads — per-RNA-position target/control logits → NLL profile loss
  5. PositionWeightedPool — learned attention pooling over RNA positions →
     binary binding classifier  (TFBindFormer: PositionWeightedPool)

Mask convention throughout: True = valid position (NOT the PyTorch
key_padding_mask convention of True = PAD).  All internal attention calls flip
the mask before passing it to nn.MultiheadAttention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Protein compressor
# ---------------------------------------------------------------------------

class ProteinCompressor(nn.Module):
    """Compress variable-length ProtT5 residues to a fixed sequence length.

    Uses target_len learned query vectors to attend over the (projected) protein
    residues.  After compression every position is a real latent token, so no
    padding mask is needed downstream.

    Args:
        protein_dim:  ProtT5 embedding dimension (1024 for full, 1024 for reduced)
        d_model:      shared hidden dimension
        target_len:   fixed output sequence length
        nhead:        attention heads for the compression step
        dropout:      dropout rate
    """

    def __init__(
        self,
        protein_dim: int = 1024,
        d_model: int = 256,
        target_len: int = 128,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_len = target_len

        self.input_proj = nn.Sequential(
            nn.Linear(protein_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.query = nn.Parameter(
            torch.randn(1, target_len, d_model) * (d_model ** -0.5)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        protein_emb: torch.Tensor,
        protein_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            protein_emb:  [B, Lp, protein_dim]
            protein_mask: [B, Lp] True = valid  (will be flipped for attention)
        Returns:
            [B, target_len, d_model]  — no padding mask needed after this
        """
        projected = self.input_proj(protein_emb)
        key_padding_mask = None if protein_mask is None else ~protein_mask.bool()
        q = self.query.expand(protein_emb.shape[0], -1, -1)
        out, _ = self.attn(
            query=q,
            key=projected,
            value=projected,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.norm(out + self.ffn(out))
        return out


# ---------------------------------------------------------------------------
# 2. Cross-attention block
# ---------------------------------------------------------------------------

class _FFNBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ffn(x))


class _CrossBlock(nn.Module):
    """One encoder layer.

    Always runs RNA→Protein attention (RNA queries protein).
    Runs Protein→RNA attention only when ``bidirectional=True``.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional

        self.rna_to_prot_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_rna = nn.LayerNorm(d_model)
        self.ffn_rna = _FFNBlock(d_model, dropout)

        if bidirectional:
            self.prot_to_rna_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.norm_prot = nn.LayerNorm(d_model)
            self.ffn_prot = _FFNBlock(d_model, dropout)

    def forward(
        self,
        rna: torch.Tensor,
        protein: torch.Tensor,
        rna_key_padding_mask: torch.Tensor | None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # RNA attends to protein (protein has no padding after compressor)
        rna_out, w_r2p = self.rna_to_prot_attn(
            query=rna,
            key=protein,
            value=protein,
            need_weights=return_weights,
            average_attn_weights=False,
        )
        rna = self.ffn_rna(self.norm_rna(rna + rna_out))

        prot_next = protein
        if self.bidirectional:
            prot_out, _ = self.prot_to_rna_attn(
                query=protein,
                key=rna,
                value=rna,
                key_padding_mask=rna_key_padding_mask,
                need_weights=False,
                average_attn_weights=False,
            )
            prot_next = self.ffn_prot(self.norm_prot(protein + prot_out))

        return rna, prot_next, w_r2p if return_weights else None


# ---------------------------------------------------------------------------
# 3. Position-weighted pooling (TFBindFormer)
# ---------------------------------------------------------------------------

class _PositionWeightedPool(nn.Module):
    """Learned attention pooling over sequence positions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        attn = torch.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * x).sum(dim=1)


# ---------------------------------------------------------------------------
# 4. Full head
# ---------------------------------------------------------------------------

class ProteinCellCrossAttentionProfileHead(nn.Module):
    """TFBindFormer-style cross-attention head on top of frozen PARNET features.

    Args:
        protein_dim:      ProtT5 residue embedding dim (typically 1024)
        rna_channels:     PARNET body_feats channels (512 for all checkpoints)
        cell_count:       number of distinct cell lines
        cell_dim:         cell embedding dimension
        d_model:          shared hidden dim for cross-attention
        nhead:            attention heads
        num_layers:       total cross-attention layers
        num_bidir_layers: how many of the first layers are bidirectional
                          (remaining layers: RNA queries protein only)
        target_prot_len:  protein compressed to this fixed length
        mix_hidden_dim:   hidden dim for mix / binding MLPs
        dropout:          dropout everywhere
    """

    def __init__(
        self,
        protein_dim: int = 1024,
        *,
        rna_channels: int = 512,
        cell_count: int = 2,
        cell_dim: int = 32,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 3,
        num_bidir_blocks: int = 2,
        target_prot_len: int = 128,
        mix_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        if num_bidir_blocks > num_blocks:
            raise ValueError("num_bidir_blocks cannot exceed num_blocks")

        self.rna_channels = rna_channels
        self.d_model = hidden_dim
        d_model = hidden_dim
        nhead = num_heads

        # Cell FiLM — applied once to RNA tokens before cross-attention
        self.cell_embedding = nn.Embedding(cell_count, cell_dim)
        self.cell_film = nn.Sequential(
            nn.Linear(cell_dim, d_model * 2),
        )
        self.rna_cell_norm = nn.LayerNorm(d_model)

        # Projections into shared d_model space
        self.rna_proj = nn.Linear(rna_channels, d_model)
        self.rna_norm = nn.LayerNorm(d_model)

        # Protein compressor: [B, Lp, protein_dim] → [B, target_prot_len, d_model]
        self.protein_compressor = ProteinCompressor(
            protein_dim=protein_dim,
            d_model=d_model,
            target_len=target_prot_len,
            nhead=nhead,
            dropout=dropout,
        )

        # Cross-attention encoder (asymmetric bidirectionality)
        self.blocks = nn.ModuleList(
            [
                _CrossBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    bidirectional=(i < num_bidir_blocks),
                )
                for i in range(num_blocks)
            ]
        )

        # Profile heads (per RNA position)
        self.target_head = nn.Linear(d_model, 1)
        self.control_head = nn.Linear(d_model, 1)
        self.mix_head = nn.Sequential(
            nn.Linear(d_model, mix_hidden_dim),
            nn.ReLU(),
            nn.Linear(mix_hidden_dim, 1),
        )

        # Binary binding head
        self.pool = _PositionWeightedPool(d_model)
        self.binding_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mix_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mix_hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    def _cell_film(self, rna: torch.Tensor, cell_index: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning from cell line to RNA tokens."""
        cell = self.cell_embedding(cell_index)
        gamma_beta = self.cell_film(cell)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        scale = 1.0 + torch.tanh(gamma)
        return self.rna_cell_norm(scale.unsqueeze(1) * rna + beta.unsqueeze(1))

    def _encode(
        self,
        rna_features: torch.Tensor,
        protein_residue_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        mask: torch.Tensor | None,
        protein_mask: torch.Tensor | None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        # RNA: channels-first [B, C, L] → tokens [B, L, d_model]
        rna = self.rna_norm(self.rna_proj(rna_features.transpose(1, 2)))
        rna = self._cell_film(rna, cell_index)

        # Protein: [B, Lp, protein_dim] → [B, target_prot_len, d_model]
        protein = self.protein_compressor(protein_residue_embedding, protein_mask)

        rna_key_padding_mask = None if mask is None else ~mask.bool()
        attn_weights: list[torch.Tensor | None] = []
        for block in self.blocks:
            rna, protein, w = block(rna, protein, rna_key_padding_mask, return_weights=return_weights)
            attn_weights.append(w)

        return rna, attn_weights

    # ------------------------------------------------------------------
    def forward(
        self,
        rna_features: torch.Tensor,
        protein_residue_embedding: torch.Tensor,
        cell_index: torch.Tensor,
        mask: torch.Tensor | None = None,
        protein_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        fused, attn_weights = self._encode(
            rna_features, protein_residue_embedding, cell_index, mask, protein_mask,
            return_weights=return_attention,
        )

        # Profile
        target_logits = self.target_head(fused).squeeze(-1)
        control_logits = self.control_head(fused).squeeze(-1)
        if mask is not None:
            fill = torch.finfo(target_logits.dtype).min
            target_logits = target_logits.masked_fill(~mask.bool(), fill)
            control_logits = control_logits.masked_fill(~mask.bool(), fill)

        target_logprob = torch.log_softmax(target_logits, dim=-1)
        control_logprob = torch.log_softmax(control_logits, dim=-1)

        pooled_mix = self._masked_mean(fused, mask)
        mix_coeff = torch.sigmoid(self.mix_head(pooled_mix)).squeeze(-1)
        mix = mix_coeff.unsqueeze(-1)
        max_lp = torch.maximum(target_logprob, control_logprob)
        total_logprob = max_lp + torch.log(
            mix * torch.exp(target_logprob - max_lp)
            + (1.0 - mix) * torch.exp(control_logprob - max_lp)
            + 1e-10
        )

        # Binary
        pooled_bind = self.pool(fused, mask)
        binding_logit = self.binding_head(pooled_bind).squeeze(-1)

        out = {
            "target_logprob": target_logprob,
            "control_logprob": control_logprob,
            "total_logprob": total_logprob,
            "target": torch.exp(target_logprob),
            "control": torch.exp(control_logprob),
            "total": torch.exp(total_logprob),
            "mix_coeff": mix_coeff,
            "binding_logit": binding_logit,
            "binding_prob": torch.sigmoid(binding_logit),
        }
        if return_attention:
            out["attn_weights"] = attn_weights
        return out

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        mask_f = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

    # ------------------------------------------------------------------
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
        profile_mask: torch.Tensor | None = None,
        binary_pos_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(
            rna_features, protein_residue_embedding, cell_index,
            mask=mask, protein_mask=protein_mask,
        )
        if mask is not None:
            count_mask = mask.to(dtype=eclip_counts.dtype)
            eclip_counts = eclip_counts * count_mask
            control_counts = control_counts * count_mask

        eclip_depth = eclip_counts.sum(dim=-1)
        control_depth = control_counts.sum(dim=-1)
        profile_keep = (eclip_depth >= min_count) if profile_mask is None else profile_mask.bool()

        eclip_nll = -(eclip_counts * out["total_logprob"]).sum(dim=-1) / eclip_depth.clamp_min(1.0)
        control_nll = -(control_counts * out["control_logprob"]).sum(dim=-1) / control_depth.clamp_min(1.0)
        n_keep = profile_keep.sum().clamp_min(1)
        profile_loss = (eclip_nll * profile_keep).sum() / n_keep
        profile_loss = profile_loss + (control_nll * profile_keep).sum() / n_keep

        binary_loss = eclip_counts.new_tensor(0.0)
        if binding_label is not None:
            pos_weight = None if binary_pos_weight is None else eclip_counts.new_tensor(binary_pos_weight)
            binary_loss = F.binary_cross_entropy_with_logits(
                out["binding_logit"],
                binding_label.to(dtype=out["binding_logit"].dtype),
                pos_weight=pos_weight,
            )

        loss = profile_loss + lambda_binary * binary_loss
        if mix_penalty:
            loss = loss + mix_penalty * out["mix_coeff"].mean()

        return {
            "loss": loss,
            "profile_loss": profile_loss,
            "binary_loss": binary_loss,
            "profile_n": profile_keep.sum(),
        }

    def loss(
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
        profile_mask: torch.Tensor | None = None,
        binary_pos_weight: float | None = None,
    ) -> torch.Tensor:
        return self.loss_components(
            rna_features, protein_residue_embedding, cell_index,
            eclip_counts, control_counts,
            binding_label=binding_label,
            mask=mask,
            protein_mask=protein_mask,
            min_count=min_count,
            mix_penalty=mix_penalty,
            lambda_binary=lambda_binary,
            profile_mask=profile_mask,
            binary_pos_weight=binary_pos_weight,
        )["loss"]


__all__ = ["ProteinCellCrossAttentionProfileHead"]
