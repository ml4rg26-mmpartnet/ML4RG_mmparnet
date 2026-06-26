#!/usr/bin/env python
"""Smoke-test the protein+cell FiLM profile head without external data.

This verifies the first multimodal model's tensor contract:

    PARNET body features [B, 512, L]
    protein embedding    [B, P]
    cell index           [B]
      -> target/control/total profiles [B, L]
"""
from __future__ import annotations

import argparse

import torch

from mmpartnet.models import ProteinCellFiLMProfileHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=600)
    parser.add_argument("--rna-channels", type=int, default=512)
    parser.add_argument("--protein-dim", type=int, default=1024)
    parser.add_argument("--cell-count", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    rna = torch.randn(args.batch_size, args.rna_channels, args.seq_len)
    protein = torch.randn(args.batch_size, args.protein_dim)
    cell_index = torch.arange(args.batch_size) % args.cell_count
    mask = torch.ones(args.batch_size, args.seq_len, dtype=torch.bool)
    eclip = torch.poisson(torch.full((args.batch_size, args.seq_len), 0.2))
    control = torch.poisson(torch.full((args.batch_size, args.seq_len), 0.2))
    binding = (torch.arange(args.batch_size) % 2 == 0).float()
    eclip[:, args.seq_len // 2] += 20
    control[:, args.seq_len // 3] += 20

    model = ProteinCellFiLMProfileHead(
        protein_dim=args.protein_dim,
        rna_channels=args.rna_channels,
        cell_count=args.cell_count,
    )
    out = model(rna, protein, cell_index, mask=mask)
    losses = model.loss_components(
        rna,
        protein,
        cell_index,
        eclip,
        control,
        binding_label=binding,
        profile_mask=binding > 0.5,
        mask=mask,
    )

    print(f"target:        {tuple(out['target'].shape)} sum={out['target'].sum(dim=-1).tolist()}")
    print(f"control:       {tuple(out['control'].shape)} sum={out['control'].sum(dim=-1).tolist()}")
    print(f"total:         {tuple(out['total'].shape)} sum={out['total'].sum(dim=-1).tolist()}")
    print(f"mix_coeff:     {tuple(out['mix_coeff'].shape)} range=({out['mix_coeff'].min():.3f}, {out['mix_coeff'].max():.3f})")
    print(f"binding_logit: {tuple(out['binding_logit'].shape)} prob_range=({out['binding_prob'].min():.3f}, {out['binding_prob'].max():.3f})")
    print(f"loss:          {float(losses['loss'].detach()):.4f}")
    print(f"profile_loss:  {float(losses['profile_loss'].detach()):.4f}")
    print(f"binary_loss:   {float(losses['binary_loss'].detach()):.4f}")

    assert out["target"].shape == (args.batch_size, args.seq_len)
    assert out["control"].shape == (args.batch_size, args.seq_len)
    assert out["total"].shape == (args.batch_size, args.seq_len)
    assert out["binding_logit"].shape == (args.batch_size,)
    assert torch.allclose(out["target"].sum(dim=-1), torch.ones(args.batch_size), atol=1e-5)
    assert torch.allclose(out["control"].sum(dim=-1), torch.ones(args.batch_size), atol=1e-5)
    assert torch.allclose(out["total"].sum(dim=-1), torch.ones(args.batch_size), atol=1e-5)
    assert torch.isfinite(losses["loss"])
    print("\nPASS: protein+cell FiLM profile head forward/loss contract is valid.")


if __name__ == "__main__":
    main()
