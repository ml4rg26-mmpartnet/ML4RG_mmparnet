#!/usr/bin/env python
"""Train the first protein+cell FiLM profile head on frozen PARNET features.

This is the minimal end-to-end multimodal experiment:

    multimodal batch -> frozen PARNET body_feats -> ProteinCellFiLMProfileHead
      -> RBPNet-style profile loss against eCLIP/control counts

By default, only exact seq_len windows are used to match PARNET pretraining.
With --include-short, shorter windows are padded and passed with a valid-position
mask; windows longer than seq_len are skipped so labels are not silently
truncated.

Modes:
  multimodal       uses RNA + protein + cell conditioning
  rna-only         zeros protein and cell conditions, leaving a global FiLM baseline
  protein-shuffle  shuffles protein embeddings within each batch while keeping RNA/cell labels fixed
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from mmpartnet.data.multimodal import (
    MultimodalCollator,
    ParnetMultimodalDataset,
    build_cell_vocab,
    load_track_protein_map,
)
from mmpartnet.models import ProteinCellFiLMProfileHead, load_parnet


SHARED = Path("/home/dgu/storage_ml4rg26-shared")
MMPARNET = Path("/home/dgu/storage_ml4rg26-mmparnet")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_HFDS = (
    SHARED
    / "parnet-eclip/data-formatted-for-training/"
    / "600nt_windows.no-one-hot.stripped/encode.filtered.hfds"
)
DEFAULT_TRACK_MAP = REPO / "mmpartnet_out/prott5_track_map.tsv"
DEFAULT_PROTEIN_H5 = MMPARNET / "manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5"
DEFAULT_OUT = REPO / "mmpartnet_out/film_runs"


def parse_tracks(value: str) -> list[int] | None:
    if value.lower() == "all":
        return None
    return [int(x) for x in value.split(",") if x.strip()]


def make_loader(
    hfds,
    split: str,
    track_map,
    track_indices: list[int] | None,
    protein_h5: Path,
    cell_to_index: dict[str, int],
    *,
    max_windows: int | None,
    seq_len: int,
    include_short: bool,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    dataset = ParnetMultimodalDataset(
        hfds[split],
        track_map,
        track_indices=track_indices,
        max_windows=max_windows,
        exact_length=None if include_short else seq_len,
        max_length=seq_len if include_short else None,
    )
    collator = MultimodalCollator(protein_h5, seq_len=seq_len, cell_to_index=cell_to_index)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def move_batch(batch: dict, device: str) -> dict:
    out = dict(batch)
    for key in ("onehot", "mask", "protein_embedding", "cell_index", "eclip", "control"):
        out[key] = batch[key].to(device)
    return out


def apply_mode(batch: dict, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    protein = batch["protein_embedding"]
    cell_index = batch["cell_index"]
    if mode == "multimodal":
        return protein, cell_index
    if mode == "rna-only":
        return torch.zeros_like(protein), torch.zeros_like(cell_index)
    if mode == "protein-shuffle":
        if protein.shape[0] <= 1:
            return protein, cell_index
        return protein[torch.randperm(protein.shape[0], device=protein.device)], cell_index
    raise ValueError(f"unknown mode {mode!r}")


def pearson_sum(
    pred: torch.Tensor,
    counts: torch.Tensor,
    min_count: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth = counts.sum(dim=-1)
    true = counts / depth.clamp_min(1.0).unsqueeze(-1)
    if mask is None:
        mask_f = torch.ones_like(pred)
    else:
        mask_f = mask.to(dtype=pred.dtype)
    valid_len = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pred_mean = (pred * mask_f).sum(dim=-1, keepdim=True) / valid_len
    true_mean = (true * mask_f).sum(dim=-1, keepdim=True) / valid_len
    pred_centered = (pred - pred_mean) * mask_f
    true_centered = (true - true_mean) * mask_f
    numerator = (pred_centered * true_centered).sum(dim=-1)
    denominator = torch.sqrt(
        (pred_centered * pred_centered).sum(dim=-1)
        * (true_centered * true_centered).sum(dim=-1)
    )
    valid = (depth >= min_count) & (denominator > 1e-9)
    corr = numerator / denominator.clamp_min(1e-9)
    return torch.where(valid, corr, torch.zeros_like(corr)).sum(), valid.sum()


def run_epoch(
    *,
    parnet,
    head: ProteinCellFiLMProfileHead,
    loader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    mode: str,
    min_count: float,
    max_batches: int | None,
    mix_penalty: float,
    progress_every: int,
) -> dict:
    training = optimizer is not None
    head.train(training)
    loss_sum = 0.0
    loss_n = 0
    pear_sum = torch.tensor(0.0, device=device)
    pear_n = torch.tensor(0, device=device)

    for step, raw_batch in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        batch = move_batch(raw_batch, device)
        protein, cell_index = apply_mode(batch, mode)
        with torch.no_grad():
            rna_features = parnet.body_feats(batch["onehot"]).detach()

        loss = head.loss(
            rna_features,
            protein,
            cell_index,
            batch["eclip"],
            batch["control"],
            mask=batch["mask"],
            min_count=min_count,
            mix_penalty=mix_penalty,
        )
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.detach().cpu())
        loss_n += 1
        with torch.no_grad():
            pred = head(rna_features, protein, cell_index, mask=batch["mask"])["total"]
            ps, pn = pearson_sum(pred, batch["eclip"], min_count, mask=batch["mask"])
            pear_sum += ps
            pear_n += pn

        if training and progress_every and step % progress_every == 0:
            pear = float((pear_sum / pear_n.clamp_min(1)).detach().cpu())
            print(f"  step={step} loss={loss_sum / loss_n:.4f} pearson={pear:+.4f} n={int(pear_n)}", flush=True)

    mean_loss = loss_sum / max(loss_n, 1)
    mean_pearson = float((pear_sum / pear_n.clamp_min(1)).detach().cpu())
    return {"loss": mean_loss, "pearson": mean_pearson, "n_profiles": int(pear_n.detach().cpu())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hfds", type=Path, default=DEFAULT_HFDS)
    parser.add_argument("--track-map", type=Path, default=DEFAULT_TRACK_MAP)
    parser.add_argument("--protein-h5", type=Path, default=DEFAULT_PROTEIN_H5)
    parser.add_argument("--tracks", default="9,138,195", help="Comma-separated track indices, or 'all'.")
    parser.add_argument("--train-split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--valid-split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--max-train-windows", type=int, default=256)
    parser.add_argument("--max-valid-windows", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=600)
    parser.add_argument(
        "--include-short",
        action="store_true",
        help="Include windows shorter than seq_len using padding/mask. Default keeps only exact seq_len windows.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-count", type=float, default=10.0)
    parser.add_argument("--mix-penalty", type=float, default=0.0)
    parser.add_argument("--mode", default="multimodal", choices=["multimodal", "rna-only", "protein-shuffle"])
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    track_indices = parse_tracks(args.tracks)
    max_train_windows = None if args.max_train_windows == 0 else args.max_train_windows
    max_valid_windows = None if args.max_valid_windows == 0 else args.max_valid_windows

    track_map = load_track_protein_map(args.track_map)
    cell_to_index = build_cell_vocab(track_map)
    hfds = load_from_disk(str(args.hfds))
    train_dataset, train_loader = make_loader(
        hfds,
        args.train_split,
        track_map,
        track_indices,
        args.protein_h5,
        cell_to_index,
        max_windows=max_train_windows,
        seq_len=args.seq_len,
        include_short=args.include_short,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataset, valid_loader = make_loader(
        hfds,
        args.valid_split,
        track_map,
        track_indices,
        args.protein_h5,
        cell_to_index,
        max_windows=max_valid_windows,
        seq_len=args.seq_len,
        include_short=args.include_short,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"device:         {device}")
    print(f"mode:           {args.mode}")
    print(f"include_short:  {args.include_short}")
    print(f"tracks:         {'all matched tracks' if track_indices is None else track_indices}")
    print(f"cell_vocab:     {cell_to_index}")
    print(f"train samples:  {len(train_dataset)}")
    print(f"valid samples:  {len(valid_dataset)}")
    print("loading frozen PARNET...", flush=True)
    parnet = load_parnet(device=device)

    probe = move_batch(next(iter(train_loader)), device)
    with torch.no_grad():
        probe_features = parnet.body_feats(probe["onehot"])
    head = ProteinCellFiLMProfileHead(
        protein_dim=int(probe["protein_embedding"].shape[1]),
        rna_channels=int(probe_features.shape[1]),
        cell_count=len(cell_to_index),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or f"film_{args.mode}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": {
            **vars(args),
            "hfds": str(args.hfds),
            "track_map": str(args.track_map),
            "protein_h5": str(args.protein_h5),
            "out_dir": str(out_dir),
            "device": device,
            "cell_to_index": cell_to_index,
            "train_samples": len(train_dataset),
            "valid_samples": len(valid_dataset),
        },
        "epochs": [],
    }

    best_pearson = float("-inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}", flush=True)
        train_stats = run_epoch(
            parnet=parnet,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mode=args.mode,
            min_count=args.min_count,
            max_batches=args.max_train_batches,
            mix_penalty=args.mix_penalty,
            progress_every=args.progress_every,
        )
        with torch.no_grad():
            valid_stats = run_epoch(
                parnet=parnet,
                head=head,
                loader=valid_loader,
                optimizer=None,
                device=device,
                mode=args.mode,
                min_count=args.min_count,
                max_batches=args.max_valid_batches,
                mix_penalty=args.mix_penalty,
                progress_every=0,
            )
        row = {"epoch": epoch, "train": train_stats, "valid": valid_stats}
        metrics["epochs"].append(row)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        print(
            f"  train loss={train_stats['loss']:.4f} pearson={train_stats['pearson']:+.4f} "
            f"n={train_stats['n_profiles']}"
        )
        print(
            f"  valid loss={valid_stats['loss']:.4f} pearson={valid_stats['pearson']:+.4f} "
            f"n={valid_stats['n_profiles']}"
        )

        if valid_stats["pearson"] > best_pearson:
            best_pearson = valid_stats["pearson"]
            torch.save(
                {
                    "model_state_dict": head.state_dict(),
                    "cell_to_index": cell_to_index,
                    "protein_dim": int(probe["protein_embedding"].shape[1]),
                    "rna_channels": int(probe_features.shape[1]),
                    "args": vars(args),
                    "epoch": epoch,
                    "valid": valid_stats,
                },
                out_dir / "best.pt",
            )
            print(f"  saved new best checkpoint: {out_dir / 'best.pt'}", flush=True)

    torch.save(head.state_dict(), out_dir / "last.statedict.pt")
    print(f"\nwrote metrics: {out_dir / 'metrics.json'}")
    print(f"wrote last:    {out_dir / 'last.statedict.pt'}")


if __name__ == "__main__":
    # Keep PARNET import stubs and torch caches out of the repo when the script is run from a shared mount.
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mmpartnet_pycache")
    main()
