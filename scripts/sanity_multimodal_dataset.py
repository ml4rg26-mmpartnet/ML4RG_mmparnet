#!/usr/bin/env python
"""Sanity-check the protein-conditioned multimodal Dataset/collator.

This does not train a model.  It verifies that one batch can be assembled as:

    RNA one-hot          [B, 4, 600]
    ProtT5 embedding     [B, 1024]
    eCLIP/control labels [B, 600]
    valid-position mask  [B, 600]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from mmpartnet.data.multimodal import (
    MultimodalCollator,
    ParnetMultimodalDataset,
    build_cell_vocab,
    load_track_protein_map,
)


SHARED = Path("/home/dgu/storage_ml4rg26-shared")
MMPARNET = Path("/home/dgu/storage_ml4rg26-mmparnet")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_HFDS = (
    SHARED
    / "parnet-eclip/data-formatted-for-training/"
    / "600nt_windows.no-one-hot.stripped/encode.filtered.hfds"
)
DEFAULT_TRACK_MAP = REPO / "mmpartnet_out/prott5_track_map.tsv"
DEFAULT_PROTEIN_H5 = (
    MMPARNET
    / "manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=600)
    parser.add_argument("--hfds", type=Path, default=DEFAULT_HFDS)
    parser.add_argument("--track-map", type=Path, default=DEFAULT_TRACK_MAP)
    parser.add_argument("--protein-h5", type=Path, default=DEFAULT_PROTEIN_H5)
    parser.add_argument(
        "--tracks",
        default="9,138,195",
        help="Comma-separated track indices to sample. Default: AQR_HepG2,QKI_HepG2,U2AF2_HepG2.",
    )
    parser.add_argument(
        "--include-short",
        action="store_true",
        help="Include short windows and use padding/mask. By default only exact seq_len windows are used.",
    )
    args = parser.parse_args()

    track_indices = [int(x) for x in args.tracks.split(",") if x.strip()]
    track_map = load_track_protein_map(args.track_map)
    cell_to_index = build_cell_vocab(track_map)
    split = load_from_disk(str(args.hfds))[args.split]
    dataset = ParnetMultimodalDataset(
        split,
        track_map,
        track_indices=track_indices,
        max_windows=args.max_windows,
        exact_length=None if args.include_short else args.seq_len,
        max_length=args.seq_len if args.include_short else None,
    )
    collator = MultimodalCollator(args.protein_h5, seq_len=args.seq_len, cell_to_index=cell_to_index)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    print(f"split:          {args.split} ({len(split)} windows before filtering)")
    print(f"dataset len:    {len(dataset)} window-track samples")
    print(f"tracks:         {track_indices}")
    print(f"onehot:         {tuple(batch['onehot'].shape)}")
    print(f"mask:           {tuple(batch['mask'].shape)} valid_nt={batch['mask'].sum(dim=1).tolist()}")
    print(f"protein:        {tuple(batch['protein_embedding'].shape)}")
    print(f"eclip:          {tuple(batch['eclip'].shape)} total_counts={batch['eclip'].sum(dim=1).tolist()}")
    print(f"control:        {tuple(batch['control'].shape)} total_counts={batch['control'].sum(dim=1).tolist()}")
    print(f"track_index:    {batch['track_index'].tolist()}")
    print(f"cell_vocab:     {cell_to_index}")
    print(f"cell_index:     {batch['cell_index'].tolist()}")
    print(f"rbp_ct:         {batch['rbp_ct']}")
    print(f"protein_h5_key: {batch['protein_h5_key']}")
    print(f"window_index:   {batch['window_index'].tolist()}")
    print(f"window_name:    {batch['window_name']}")

    assert batch["onehot"].shape[1:] == (4, args.seq_len)
    assert batch["mask"].shape[1:] == (args.seq_len,)
    assert batch["protein_embedding"].shape[1] == 1024
    assert batch["cell_index"].shape == batch["track_index"].shape
    assert batch["eclip"].shape[1:] == (args.seq_len,)
    assert batch["control"].shape[1:] == (args.seq_len,)
    assert torch.isfinite(batch["protein_embedding"]).all()
    print("\nPASS: multimodal batch assembled correctly.")


if __name__ == "__main__":
    main()
