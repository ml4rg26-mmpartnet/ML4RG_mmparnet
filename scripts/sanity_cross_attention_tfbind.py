#!/usr/bin/env python
"""Smoke-test the TFBindFormer-style cross-attention profile head.

This verifies our rebuilt model's tensor contract (kept separate from the
original ProteinCellCrossAttentionProfileHead so both can be compared):

    PARNET body features     [B, C, L]
    ProtT5 residue embedding [B, Lp, P]
    protein mask             [B, Lp]
    cell index               [B]
      -> target/control/total profiles [B, L]
      -> binding_logit                 [B]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from mmpartnet.models import TFBindCrossAttentionProfileHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=600)
    parser.add_argument("--protein-len", type=int, default=350)
    parser.add_argument("--rna-channels", type=int, default=512)
    parser.add_argument("--protein-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--target-prot-len", type=int, default=128)
    parser.add_argument("--cell-count", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = args.device
    rna = torch.randn(args.batch_size, args.rna_channels, args.seq_len, device=dev)
    protein = torch.randn(args.batch_size, args.protein_len, args.protein_dim, device=dev)
    protein_mask = torch.ones(args.batch_size, args.protein_len, dtype=torch.bool, device=dev)
    protein_mask[-1, args.protein_len // 2 :] = False
    cell_index = torch.arange(args.batch_size, device=dev) % args.cell_count
    mask = torch.ones(args.batch_size, args.seq_len, dtype=torch.bool, device=dev)
    mask[-1, args.seq_len - 25 :] = False
    eclip = torch.poisson(torch.full((args.batch_size, args.seq_len), 0.2)).to(dev)
    control = torch.poisson(torch.full((args.batch_size, args.seq_len), 0.2)).to(dev)
    binding = (torch.arange(args.batch_size, device=dev) % 2 == 0).float()
    eclip[:, args.seq_len // 2] += 20
    control[:, args.seq_len // 3] += 20

    model = TFBindCrossAttentionProfileHead(
        protein_dim=args.protein_dim,
        rna_channels=args.rna_channels,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        target_prot_len=args.target_prot_len,
        cell_count=args.cell_count,
    ).to(dev)

    out = model(rna, protein, cell_index, mask=mask, protein_mask=protein_mask)
    losses = model.loss_components(
        rna,
        protein,
        cell_index,
        eclip,
        control,
        binding_label=binding,
        profile_mask=binding > 0.5,
        mask=mask,
        protein_mask=protein_mask,
    )
    losses["loss"].backward()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params:        {n_params / 1e6:.2f}M")
    print(f"target:        {tuple(out['target'].shape)} sum={out['target'].sum(dim=-1).tolist()}")
    print(f"control:       {tuple(out['control'].shape)} sum={out['control'].sum(dim=-1).tolist()}")
    print(f"total:         {tuple(out['total'].shape)} sum={out['total'].sum(dim=-1).tolist()}")
    print(f"mix_coeff:     {tuple(out['mix_coeff'].shape)} range=({out['mix_coeff'].min():.3f}, {out['mix_coeff'].max():.3f})")
    print(f"binding_logit: {tuple(out['binding_logit'].shape)} prob_range=({out['binding_prob'].min():.3f}, {out['binding_prob'].max():.3f})")
    print(f"loss:          {float(losses['loss'].detach()):.4f}")
    print(f"profile_loss:  {float(losses['profile_loss'].detach()):.4f}")
    print(f"binary_loss:   {float(losses['binary_loss'].detach()):.4f}")

    B, L = args.batch_size, args.seq_len
    assert out["target"].shape == (B, L)
    assert out["control"].shape == (B, L)
    assert out["total"].shape == (B, L)
    assert out["binding_logit"].shape == (B,)
    ones = torch.ones(B, device=dev)
    assert torch.allclose(out["target"].sum(dim=-1), ones, atol=1e-5)
    assert torch.allclose(out["control"].sum(dim=-1), ones, atol=1e-5)
    assert torch.allclose(out["total"].sum(dim=-1), ones, atol=1e-5)
    assert torch.isfinite(losses["loss"])
    print("\nPASS: TFBind cross-attention forward/backward/loss contract is valid.")


if __name__ == "__main__":
    main()
