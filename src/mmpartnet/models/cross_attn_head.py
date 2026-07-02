"""Cross-attention multimodal binding heads (TFBindFormer-analogous). The protein representation
(ESM/ProtT5, pooled OR per-residue) cross-attends with the RNA window's per-position PARNET features, so
the protein SELECTS which positions matter -- a richer conditioning than the static concat/mean-max pool of
EarlyFusion/ConditionedHead. Returns a per-(window, protein) binding logit.

Two heads, two directions:
  * CrossAttnHead   -- pooled protein -> single query token cross-attends over RNA positions (K/V).
                       The protein->positions direction; the cross-attention map (1,P) says WHICH RNA
                       positions the protein selects, so it is directly comparable to an ISM/IG
                       attribution map over the same window (the architecture's built-in interpretability).
  * BiCrossAttnHead -- per-residue protein (Lp tokens) as K/V; RNA positions as queries attend over
                       protein residues, then pool. The faithful TFBindFormer analog (TF protein tokens
                       x DNA), so the attention map (P, Lp) says which protein RESIDUES drive binding at
                       which RNA positions -- a domain-level (RRM/KH) readout the pooled head cannot give.

`return_attn=True` returns (logit, attn) where attn is the layer-mean cross-attention weights, so any
caller can pull the map for the attention-vs-attribution faithfulness check without retraining.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CrossAttnHead(nn.Module):
    """Pooled protein query x RNA per-position features. attn map: (B, 1, P)."""

    def __init__(self, d_model: int = 512, dp: int = 1280, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, use_pos: bool = True):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(dp, d_model), nn.LayerNorm(d_model))
        self.use_pos = use_pos
        self.pos = nn.Parameter(torch.zeros(1, 1, d_model))                     # learned query seed offset
        self.attn = nn.ModuleList([nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
                                   for _ in range(layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                                nn.Dropout(dropout), nn.Linear(d_model, d_model))
                                  for _ in range(layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, H, p, return_attn: bool = False):
        """H: (B, P, d_model) RNA per-position features; p: (B, dp) protein embedding -> (B,) binding logit."""
        x = self.q_proj(p).unsqueeze(1)                                        # (B, 1, d_model) protein query
        if self.use_pos:
            x = x + self.pos
        amaps = []
        for attn, n1, ffn, n2 in zip(self.attn, self.norm1, self.ffn, self.norm2):
            ctx, aw = attn(x, H, H, need_weights=return_attn, average_attn_weights=True)
            x = n1(x + ctx)
            x = n2(x + ffn(x))
            if return_attn:
                amaps.append(aw)                                              # (B, 1, P)
        logit = self.out(x.squeeze(1)).squeeze(-1)
        if return_attn:
            return logit, torch.stack(amaps).mean(0).squeeze(1)              # (B, P) layer-mean RNA attention
        return logit


class BiCrossAttnHead(nn.Module):
    """Per-residue protein K/V (the faithful TFBindFormer analog). RNA per-position features are projected
    to queries that attend over the protein residue tokens; the attended per-position context is pooled to
    a binding logit. attn map: (B, P, Lp) = which protein residues each RNA position selects.
    A learned [BIND] query summarizes the per-position context (so the pooled read does not wash it out)."""

    def __init__(self, d_model: int = 512, dp: int = 1280, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, d_prot: int = 256):
        super().__init__()
        self.rna_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.prot_proj = nn.Sequential(nn.Linear(dp, d_model), nn.LayerNorm(d_model))   # residue tokens
        self.attn = nn.ModuleList([nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
                                   for _ in range(layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                                nn.Dropout(dropout), nn.Linear(d_model, d_model))
                                  for _ in range(layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.bind = nn.Parameter(torch.zeros(1, 1, d_model))                  # [BIND] pooling query
        self.pool = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, H, P, mask=None, return_attn: bool = False):
        """H: (B, Prna, d_model) RNA per-position feats; P: (B, Lp, dp) protein per-residue;
        mask: (B, Lp) bool, True=pad. -> (B,) logit."""
        x = self.rna_proj(H)                                                  # (B, Prna, d_model) queries
        kv = self.prot_proj(P)                                                # (B, Lp, d_model) prot residues
        amaps = []
        for attn, n1, ffn, n2 in zip(self.attn, self.norm1, self.ffn, self.norm2):
            ctx, aw = attn(x, kv, kv, key_padding_mask=mask,
                           need_weights=return_attn, average_attn_weights=True)
            x = n1(x + ctx)
            x = n2(x + ffn(x))
            if return_attn:
                amaps.append(aw)                                              # (B, Prna, Lp)
        b = self.bind.expand(x.size(0), -1, -1)                              # (B,1,d)
        pooled, _ = self.pool(b, x, x)                                        # attend over RNA positions
        logit = self.out(pooled.squeeze(1)).squeeze(-1)
        if return_attn:
            return logit, torch.stack(amaps).mean(0)                          # (B, Prna, Lp)
        return logit


class BidirCrossAttnHead(nn.Module):
    """Fully bidirectional fused head (the complete TFBindFormer/CORAL block): each layer updates BOTH streams
    -- RNA positions attend protein residues AND protein residues attend RNA positions -- then a [BIND] query
    pools the RNA stream and the (masked-mean) protein stream are concatenated to a logit. Exposes both maps:
    rna->prot (which protein residues / domains) and prot->rna (which RNA positions)."""

    def __init__(self, d_model: int = 512, dp: int = 1280, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.rna_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.prot_proj = nn.Sequential(nn.Linear(dp, d_model), nn.LayerNorm(d_model))
        self.a_rp = nn.ModuleList([nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True) for _ in range(layers)])
        self.a_pr = nn.ModuleList([nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True) for _ in range(layers)])
        self.nr = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.np_ = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.fr = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model)) for _ in range(layers)])
        self.fp = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model)) for _ in range(layers)])
        self.nr2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.np2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.bind = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, H, P, mask=None, return_attn: bool = False):
        r = self.rna_proj(H); p = self.prot_proj(P)
        rp_maps = []
        for arp, apr, nr, np_, fr, fp, nr2, np2 in zip(self.a_rp, self.a_pr, self.nr, self.np_, self.fr, self.fp, self.nr2, self.np2):
            r_ctx, aw = arp(r, p, p, key_padding_mask=mask, need_weights=return_attn, average_attn_weights=True)  # RNA<-prot
            p_ctx, _ = apr(p, r, r, need_weights=False)                       # prot<-RNA
            r = nr(r + r_ctx); p = np_(p + p_ctx)
            r = nr2(r + fr(r)); p = np2(p + fp(p))
            if return_attn:
                rp_maps.append(aw)                                            # (B,Prna,Lp)
        b = self.bind.expand(r.size(0), -1, -1)
        r_vec, _ = self.pool(b, r, r); r_vec = r_vec.squeeze(1)
        if mask is not None:
            w = (~mask).float().unsqueeze(-1); p_vec = (p * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            p_vec = p.mean(1)
        logit = self.out(torch.cat([r_vec, p_vec], -1)).squeeze(-1)
        if return_attn:
            return logit, torch.stack(rp_maps).mean(0)
        return logit
