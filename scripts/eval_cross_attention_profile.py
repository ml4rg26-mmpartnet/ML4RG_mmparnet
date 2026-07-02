#!/usr/bin/env python
"""Evaluate a trained protein-residue cross-attention checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from datasets import load_from_disk

from mmpartnet.data.multimodal import build_cell_vocab, load_track_protein_map
from mmpartnet.models import ProteinCellCrossAttentionProfileHead, load_parnet
from scripts.train_cross_attention_profile import (
    DEFAULT_BINDING,
    DEFAULT_HFDS,
    DEFAULT_PROTEIN_H5,
    DEFAULT_TRACK_MAP,
    make_loader,
    parse_tracks,
    run_epoch,
)


def checkpoint_args(checkpoint: dict) -> dict:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def arg_or_checkpoint(cli_value, ckpt_args: dict, name: str, default=None):
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full checkpoint, usually best_pearson.pt.")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--hfds", type=Path, default=None)
    parser.add_argument("--binding-dataset", type=Path, default=None)
    parser.add_argument("--track-map", type=Path, default=None)
    parser.add_argument("--protein-h5", type=Path, default=None)
    parser.add_argument("--tracks", default=None, help="Comma-separated track indices, 'all', or checkpoint value.")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--include-short", action="store_true", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-protein-len", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=[None, "multimodal", "rna-only", "protein-shuffle", "no-cell"])
    parser.add_argument("--min-count", type=float, default=None)
    parser.add_argument("--mix-penalty", type=float, default=None)
    parser.add_argument("--lambda-profile", type=float, default=None)
    parser.add_argument("--lambda-binary", type=float, default=None)
    parser.add_argument("--binary-pos-weight", type=float, default=None)
    parser.add_argument(
        "--profile-mask-source",
        default=None,
        choices=[None, "binding", "count", "binding-and-count"],
    )
    parser.add_argument("--task", default=None, choices=[None, "multitask", "profile-only", "binary-only"])
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON path for evaluation metrics.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint_args(checkpoint)

    hfds_path = Path(arg_or_checkpoint(args.hfds, ckpt_args, "hfds", DEFAULT_HFDS))
    binding_path = Path(arg_or_checkpoint(args.binding_dataset, ckpt_args, "binding_dataset", DEFAULT_BINDING))
    track_map_path = Path(arg_or_checkpoint(args.track_map, ckpt_args, "track_map", DEFAULT_TRACK_MAP))
    protein_h5 = Path(arg_or_checkpoint(args.protein_h5, ckpt_args, "protein_h5", DEFAULT_PROTEIN_H5))
    tracks_value = arg_or_checkpoint(args.tracks, ckpt_args, "tracks", "all")
    max_windows_raw = arg_or_checkpoint(args.max_windows, ckpt_args, f"max_{args.split}_windows", 0)
    max_windows = None if int(max_windows_raw) == 0 else int(max_windows_raw)
    seq_len = int(arg_or_checkpoint(args.seq_len, ckpt_args, "seq_len", 600))
    include_short = bool(arg_or_checkpoint(args.include_short, ckpt_args, "include_short", False))
    batch_size = int(arg_or_checkpoint(args.batch_size, ckpt_args, "batch_size", 8))
    max_protein_len = arg_or_checkpoint(args.max_protein_len, ckpt_args, "max_protein_len", None)
    if max_protein_len is not None:
        max_protein_len = int(max_protein_len)
    mode = arg_or_checkpoint(args.mode, ckpt_args, "mode", "multimodal")
    min_count = float(arg_or_checkpoint(args.min_count, ckpt_args, "min_count", 10.0))
    mix_penalty = float(arg_or_checkpoint(args.mix_penalty, ckpt_args, "mix_penalty", 0.0))
    lambda_profile = float(arg_or_checkpoint(args.lambda_profile, ckpt_args, "lambda_profile", 1.0))
    lambda_binary = float(arg_or_checkpoint(args.lambda_binary, ckpt_args, "lambda_binary", 1.0))
    binary_pos_weight = arg_or_checkpoint(args.binary_pos_weight, ckpt_args, "binary_pos_weight", None)
    profile_mask_source = arg_or_checkpoint(args.profile_mask_source, ckpt_args, "profile_mask_source", "binding")
    task = arg_or_checkpoint(args.task, ckpt_args, "task", "multitask")

    track_indices = parse_tracks(str(tracks_value))
    track_map = load_track_protein_map(track_map_path)
    cell_to_index = checkpoint.get("cell_to_index") or build_cell_vocab(track_map)
    hfds = load_from_disk(str(hfds_path))
    binding_data = torch.load(binding_path, map_location="cpu", weights_only=False)
    dataset, loader = make_loader(
        hfds,
        binding_data,
        args.split,
        track_map,
        track_indices,
        protein_h5,
        cell_to_index,
        max_windows=max_windows,
        seq_len=seq_len,
        include_short=include_short,
        batch_size=batch_size,
        shuffle=False,
        balanced=False,
        balanced_pos_fraction=0.5,
        steps_per_epoch=None,
        seed=0,
        num_workers=args.num_workers,
        max_protein_len=max_protein_len,
    )

    print(f"checkpoint:     {args.checkpoint}")
    print(f"device:         {device}")
    print(f"split:          {args.split}")
    print(f"task:           {task}")
    print(f"mode:           {mode}")
    print(f"max_protein_len:{max_protein_len}")
    print(f"tracks:         {'all matched tracks' if track_indices is None else track_indices}")
    print(f"samples:        {len(dataset)}")
    print("loading frozen PARNET...", flush=True)
    parnet = load_parnet(device=device)

    config = checkpoint.get("model_config", {})
    head = ProteinCellCrossAttentionProfileHead(
        protein_dim=int(checkpoint["protein_dim"]),
        rna_channels=int(checkpoint["rna_channels"]),
        cell_count=len(cell_to_index),
        cell_dim=int(config.get("cell_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_heads=int(config.get("num_heads", 8)),
        num_blocks=int(config.get("num_blocks", 1)),
        dropout=float(config.get("dropout", 0.1)),
        protein_projection_hidden_dim=int(config.get("protein_projection_hidden_dim", 0)),
        protein_compression=str(config.get("protein_compression", "none")),
        protein_latent_len=int(config.get("protein_latent_len", 256)),
        binary_pooling=str(config.get("binary_pooling", "position")),
        binary_alpha_source=str(config.get("binary_alpha_source", "gated")),
    ).to(device)
    head.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        stats = run_epoch(
            parnet=parnet,
            head=head,
            loader=loader,
            optimizer=None,
            device=device,
            mode=mode,
            min_count=min_count,
            max_batches=args.max_batches,
            mix_penalty=mix_penalty,
            lambda_profile=lambda_profile,
            lambda_binary=lambda_binary,
            binary_pos_weight=binary_pos_weight,
            profile_mask_source=profile_mask_source,
            task=task,
            progress_every=0,
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "config": {
            "hfds": str(hfds_path),
            "binding_dataset": str(binding_path),
            "track_map": str(track_map_path),
            "protein_h5": str(protein_h5),
            "tracks": tracks_value,
            "max_windows": max_windows,
            "batch_size": batch_size,
            "max_batches": args.max_batches,
            "max_protein_len": max_protein_len,
            "task": task,
            "mode": mode,
            "lambda_profile": lambda_profile,
            "lambda_binary": lambda_binary,
            "profile_mask_source": profile_mask_source,
        },
        "metrics": stats,
    }
    print(json.dumps(result["metrics"], indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote eval:     {args.out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mmpartnet_pycache")
    main()
