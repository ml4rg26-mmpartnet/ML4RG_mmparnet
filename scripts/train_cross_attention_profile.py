#!/usr/bin/env python
"""Train a protein-residue cross-attention profile head on frozen PARNET features.

The model uses RNA positions from the frozen PARNET body as queries and padded
ProtT5 residue embeddings as keys/values. It keeps the FiLM baseline's
multitask objective, sampler, validation metrics, and checkpoint names so runs
can be compared directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Sampler

from mmpartnet.data.multimodal import (
    MultimodalCollator,
    ParnetMultimodalDataset,
    build_cell_vocab,
    load_track_protein_map,
)
from mmpartnet.models import load_parnet
from mmpartnet.models.cross_attention_dgu import ProteinCellCrossAttentionProfileHead

from scripts.train_film_profile import (
    BindingBalancedSampler,
    DEFAULT_HFDS,
    DEFAULT_TRACK_MAP,
    binary_average_precision,
    binary_stats,
    parse_tracks,
    pearson_sum,
)


MMPARNET = Path("/home/dgu/storage_ml4rg26-mmparnet")
DEFAULT_PROTEIN_H5 = MMPARNET / "manually_gathered/ProtT5_zenodo_datasets/embeddings_file.h5"
DEFAULT_BINDING = (
    MMPARNET
    / "manually_gathered/600nt_windows.no-one-hot.stripped.binding/"
    / "600nt_windows.no-one-hot.stripped.binding.pureclip/dataset.pt"
)
DEFAULT_OUT = REPO / "mmpartnet_out/cross_attention_runs"


class FixedSubsetSampler(Sampler[int]):
    """Iterate a fixed, deterministic subset of flattened dataset indices."""

    def __init__(self, dataset_size: int, *, num_samples: int, seed: int):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        keep = min(int(num_samples), int(dataset_size))
        self.indices = torch.randperm(int(dataset_size), generator=generator)[:keep].tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def make_loader(
    hfds,
    binding_data,
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
    balanced: bool,
    balanced_pos_fraction: float,
    steps_per_epoch: int | None,
    seed: int,
    num_workers: int,
    max_protein_len: int | None,
    sample_size: int | None = None,
    sample_seed: int = 0,
):
    dataset = ParnetMultimodalDataset(
        hfds[split],
        track_map,
        binding_split=None if binding_data is None else binding_data[split],
        track_indices=track_indices,
        max_windows=max_windows,
        exact_length=None if include_short else seq_len,
        max_length=seq_len if include_short else None,
    )
    sampler = None
    if balanced:
        sampler_steps = steps_per_epoch or 1000
        sampler = BindingBalancedSampler(
            dataset,
            positive_fraction=balanced_pos_fraction,
            num_samples=sampler_steps * batch_size,
            seed=seed,
        )
        shuffle = False
    elif sample_size is not None:
        sampler = FixedSubsetSampler(len(dataset), num_samples=sample_size, seed=sample_seed)
        shuffle = False
    collator = MultimodalCollator(
        protein_h5,
        seq_len=seq_len,
        cell_to_index=cell_to_index,
        return_residue_embeddings=True,
        max_protein_len=max_protein_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def move_batch(batch: dict, device: str) -> dict:
    out = dict(batch)
    for key in (
        "onehot",
        "mask",
        "protein_embedding",
        "protein_residue_embedding",
        "protein_mask",
        "cell_index",
        "eclip",
        "control",
    ):
        out[key] = batch[key].to(device)
    if "binding" in batch:
        out["binding"] = batch["binding"].to(device)
    return out


def apply_mode(batch: dict, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    protein = batch["protein_residue_embedding"]
    protein_mask = batch["protein_mask"]
    cell_index = batch["cell_index"]
    if mode == "multimodal":
        return protein, protein_mask, cell_index
    if mode == "rna-only":
        return torch.zeros_like(protein), protein_mask, torch.full_like(cell_index, -1)
    if mode == "no-cell":
        return protein, protein_mask, torch.full_like(cell_index, -1)
    if mode == "protein-shuffle":
        if protein.shape[0] <= 1:
            return protein, protein_mask, cell_index
        order = torch.randperm(protein.shape[0], device=protein.device)
        return protein[order], protein_mask[order], cell_index
    raise ValueError(f"unknown mode {mode!r}")


def distribution_entropy(prob: torch.Tensor) -> torch.Tensor:
    """Mean entropy of a batch of position distributions."""
    return -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1).mean()


def distribution_max(prob: torch.Tensor) -> torch.Tensor:
    """Mean max position probability of a batch of position distributions."""
    return prob.max(dim=-1).values.mean()


def distribution_topk_mass(prob: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Mean probability mass in the top-k positions."""
    k = min(k, prob.shape[-1])
    return prob.topk(k, dim=-1).values.sum(dim=-1).mean()


def save_training_checkpoint(
    path: Path,
    *,
    head: ProteinCellCrossAttentionProfileHead,
    optimizer: torch.optim.Optimizer,
    cell_to_index: dict[str, int],
    protein_dim: int,
    rna_channels: int,
    args: argparse.Namespace,
    epoch: int,
    best_pearson: float,
    best_auprc: float,
    metrics: dict,
    valid_stats: dict | None = None,
) -> None:
    torch.save(
        {
            "model_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cell_to_index": cell_to_index,
            "protein_dim": protein_dim,
            "rna_channels": rna_channels,
            "model_type": "cross_attention",
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_blocks": args.num_blocks,
                "cell_dim": args.cell_dim,
                "dropout": args.dropout,
                "protein_projection_hidden_dim": args.protein_projection_hidden_dim,
                "protein_compression": args.protein_compression,
                "protein_latent_len": args.protein_latent_len,
                "binary_pooling": args.binary_pooling,
                "binary_alpha_source": args.binary_alpha_source,
            },
            "args": vars(args),
            "epoch": epoch,
            "best_pearson": best_pearson,
            "best_auprc": best_auprc,
            "metrics": metrics,
            "valid": valid_stats,
        },
        path,
    )


def run_epoch(
    *,
    parnet,
    head: ProteinCellCrossAttentionProfileHead,
    loader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    mode: str,
    min_count: float,
    max_batches: int | None,
    mix_penalty: float,
    lambda_profile: float,
    lambda_binary: float,
    binary_pos_weight: float | None,
    profile_mask_source: str,
    task: str,
    progress_every: int,
) -> dict:
    training = optimizer is not None
    head.train(training)
    loss_sum = 0.0
    loss_n = 0
    pear_sum = torch.tensor(0.0, device=device)
    pear_n = torch.tensor(0, device=device)
    profile_loss_sum = 0.0
    binary_loss_sum = 0.0
    profile_mask_n = 0
    binding_n = 0
    binding_pos = 0
    binding_correct = 0
    binary_scores = []
    binary_labels = []
    gate_sum = 0.0
    gate_sq_sum = 0.0
    gate_n = 0
    gate_pos_sum = 0.0
    gate_pos_n = 0
    gate_neg_sum = 0.0
    gate_neg_n = 0
    target_entropy_sum = 0.0
    binary_entropy_sum = 0.0
    alpha_entropy_sum = 0.0
    target_max_sum = 0.0
    binary_max_sum = 0.0
    alpha_max_sum = 0.0
    target_top10_sum = 0.0
    binary_top10_sum = 0.0
    alpha_top10_sum = 0.0
    attention_summary_n = 0

    for step, raw_batch in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        batch = move_batch(raw_batch, device)
        protein, protein_mask, cell_index = apply_mode(batch, mode)
        with torch.no_grad():
            rna_features = parnet.body_feats(batch["onehot"]).detach()

        binding = batch.get("binding")
        if profile_mask_source == "binding":
            profile_mask = None if binding is None else binding > 0.5
        elif profile_mask_source == "count":
            profile_mask = batch["eclip"].sum(dim=-1) >= min_count
        elif profile_mask_source == "binding-and-count":
            count_mask = batch["eclip"].sum(dim=-1) >= min_count
            profile_mask = count_mask if binding is None else ((binding > 0.5) & count_mask)
        else:
            raise ValueError(f"unknown profile_mask_source {profile_mask_source!r}")

        losses = head.loss_components(
            rna_features,
            protein,
            cell_index,
            batch["eclip"],
            batch["control"],
            binding_label=binding,
            mask=batch["mask"],
            protein_mask=protein_mask,
            min_count=min_count,
            mix_penalty=mix_penalty,
            lambda_profile=lambda_profile,
            lambda_binary=lambda_binary,
            profile_mask=profile_mask,
            binary_pos_weight=binary_pos_weight,
            task=task,
        )
        loss = losses["loss"]
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.detach().cpu())
        profile_loss_sum += float(losses["profile_loss"].detach().cpu())
        binary_loss_sum += float(losses["binary_loss"].detach().cpu())
        profile_mask_n += int(losses["profile_n"].detach().cpu())
        loss_n += 1
        with torch.no_grad():
            out = head(rna_features, protein, cell_index, mask=batch["mask"], protein_mask=protein_mask, task=task)
            if "total" in out:
                ps, pn = pearson_sum(out["total"], batch["eclip"], min_count, mask=batch["mask"], valid_mask=profile_mask)
                pear_sum += ps
                pear_n += pn
            if "binding_gate" in out:
                gate = out["binding_gate"].detach()
                gate_sum += float(gate.sum().cpu())
                gate_sq_sum += float((gate * gate).sum().cpu())
                gate_n += int(gate.numel())
                if binding is not None:
                    pos_mask = binding.detach() > 0.5
                    neg_mask = ~pos_mask
                    if bool(pos_mask.any()):
                        gate_pos_sum += float(gate[pos_mask].sum().cpu())
                        gate_pos_n += int(pos_mask.sum().cpu())
                    if bool(neg_mask.any()):
                        gate_neg_sum += float(gate[neg_mask].sum().cpu())
                        gate_neg_n += int(neg_mask.sum().cpu())
            if "target" in out:
                target_prob = out["target"].detach()
                target_entropy_sum += float(distribution_entropy(target_prob).cpu())
                target_max_sum += float(distribution_max(target_prob).cpu())
                target_top10_sum += float(distribution_topk_mass(target_prob, k=10).cpu())
            if "binary_position_prob" in out:
                binary_prob = out["binary_position_prob"].detach()
                binary_entropy_sum += float(distribution_entropy(binary_prob).cpu())
                binary_max_sum += float(distribution_max(binary_prob).cpu())
                binary_top10_sum += float(distribution_topk_mass(binary_prob, k=10).cpu())
            if "alpha_bind" in out:
                alpha_prob = out["alpha_bind"].detach()
                alpha_entropy_sum += float(distribution_entropy(alpha_prob).cpu())
                alpha_max_sum += float(distribution_max(alpha_prob).cpu())
                alpha_top10_sum += float(distribution_topk_mass(alpha_prob, k=10).cpu())
            if "target" in out or "binary_position_prob" in out or "alpha_bind" in out:
                attention_summary_n += 1
            if "binding_logit" in out and binding is not None:
                bs = binary_stats(out["binding_logit"], binding)
                binding_n += bs["n"]
                binding_pos += bs["pos"]
                binding_correct += bs["accuracy_sum"]
                binary_scores.append(torch.sigmoid(out["binding_logit"].detach()).cpu())
                binary_labels.append(binding.detach().cpu())

        if training and progress_every and step % progress_every == 0:
            pear = float((pear_sum / pear_n.clamp_min(1)).detach().cpu())
            bind_acc = binding_correct / max(binding_n, 1)
            print(
                f"  step={step} loss={loss_sum / loss_n:.4f} "
                f"profile={profile_loss_sum / loss_n:.4f} binary={binary_loss_sum / loss_n:.4f} "
                f"pearson={pear:+.4f} n={int(pear_n)} bind_acc={bind_acc:.3f}",
                flush=True,
            )

    mean_loss = loss_sum / max(loss_n, 1)
    mean_pearson = float((pear_sum / pear_n.clamp_min(1)).detach().cpu())
    if binary_scores:
        binding_auprc = binary_average_precision(torch.cat(binary_scores), torch.cat(binary_labels))
    else:
        binding_auprc = 0.0
    return {
        "loss": mean_loss,
        "profile_loss": profile_loss_sum / max(loss_n, 1),
        "binary_loss": binary_loss_sum / max(loss_n, 1),
        "pearson": mean_pearson,
        "n_profiles": int(pear_n.detach().cpu()),
        "profile_mask_n": profile_mask_n,
        "binding_n": binding_n,
        "binding_pos": binding_pos,
        "binding_pos_rate": binding_pos / max(binding_n, 1),
        "binding_accuracy": binding_correct / max(binding_n, 1),
        "binding_auprc": binding_auprc,
        "binding_gate_mean": gate_sum / max(gate_n, 1),
        "binding_gate_std": max(gate_sq_sum / max(gate_n, 1) - (gate_sum / max(gate_n, 1)) ** 2, 0.0) ** 0.5,
        "binding_gate_pos_mean": gate_pos_sum / max(gate_pos_n, 1),
        "binding_gate_neg_mean": gate_neg_sum / max(gate_neg_n, 1),
        "target_entropy": target_entropy_sum / max(attention_summary_n, 1),
        "binary_position_entropy": binary_entropy_sum / max(attention_summary_n, 1),
        "alpha_bind_entropy": alpha_entropy_sum / max(attention_summary_n, 1),
        "target_max_prob": target_max_sum / max(attention_summary_n, 1),
        "binary_position_max_prob": binary_max_sum / max(attention_summary_n, 1),
        "alpha_bind_max_prob": alpha_max_sum / max(attention_summary_n, 1),
        "target_top10_mass": target_top10_sum / max(attention_summary_n, 1),
        "binary_position_top10_mass": binary_top10_sum / max(attention_summary_n, 1),
        "alpha_bind_top10_mass": alpha_top10_sum / max(attention_summary_n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hfds", type=Path, default=DEFAULT_HFDS)
    parser.add_argument("--binding-dataset", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--track-map", type=Path, default=DEFAULT_TRACK_MAP)
    parser.add_argument("--protein-h5", type=Path, default=DEFAULT_PROTEIN_H5)
    parser.add_argument("--tracks", default="9,138,195", help="Comma-separated track indices, or 'all'.")
    parser.add_argument("--train-split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--valid-split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--max-train-windows", type=int, default=256)
    parser.add_argument("--max-valid-windows", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=600)
    parser.add_argument("--include-short", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--balanced-train", action="store_true")
    parser.add_argument("--balanced-pos-fraction", type=float, default=0.5)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--cell-dim", type=int, default=32)
    parser.add_argument("--max-protein-len", type=int, default=None)
    parser.add_argument(
        "--protein-projection-hidden-dim",
        type=int,
        default=768,
        help="Middle dimension for protein projection MLP. Use 0 for the old single Linear projection.",
    )
    parser.add_argument("--protein-compression", default="latent", choices=["none", "latent"])
    parser.add_argument("--protein-latent-len", type=int, default=256)
    parser.add_argument(
        "--binary-pooling",
        default="position",
        choices=["position", "mean"],
        help="Pooling used by the binary-only binding head. Multitask keeps gated target/position pooling.",
    )
    parser.add_argument(
        "--binary-alpha-source",
        default="gated",
        choices=["gated", "target", "target-detached", "binary"],
        help="Position distribution used by the multitask binary head.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-count", type=float, default=10.0)
    parser.add_argument("--mix-penalty", type=float, default=0.0)
    parser.add_argument("--lambda-profile", type=float, default=1.0)
    parser.add_argument("--lambda-binary", type=float, default=20.0)
    parser.add_argument("--binary-pos-weight", type=float, default=None)
    parser.add_argument(
        "--profile-mask-source",
        default="binding",
        choices=["binding", "count", "binding-and-count"],
    )
    parser.add_argument("--task", default="multitask", choices=["multitask", "profile-only", "binary-only"])
    parser.add_argument("--mode", default="multimodal", choices=["multimodal", "rna-only", "protein-shuffle", "no-cell"])
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument(
        "--valid-sample-size",
        type=int,
        default=None,
        help="Use a fixed random subset of this many flattened validation samples.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    track_indices = parse_tracks(args.tracks)
    max_train_windows = None if args.max_train_windows == 0 else args.max_train_windows
    max_valid_windows = None if args.max_valid_windows == 0 else args.max_valid_windows

    track_map = load_track_protein_map(args.track_map)
    cell_to_index = build_cell_vocab(track_map)
    hfds = load_from_disk(str(args.hfds))
    binding_data = torch.load(args.binding_dataset, map_location="cpu", weights_only=False)
    train_dataset, train_loader = make_loader(
        hfds,
        binding_data,
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
        balanced=args.balanced_train,
        balanced_pos_fraction=args.balanced_pos_fraction,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        num_workers=args.num_workers,
        max_protein_len=args.max_protein_len,
    )
    valid_dataset, valid_loader = make_loader(
        hfds,
        binding_data,
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
        balanced=False,
        balanced_pos_fraction=args.balanced_pos_fraction,
        steps_per_epoch=None,
        seed=args.seed,
        num_workers=args.num_workers,
        max_protein_len=args.max_protein_len,
        sample_size=args.valid_sample_size,
        sample_seed=args.seed,
    )

    print(f"device:         {device}")
    print(f"task:           {args.task}")
    print(f"mode:           {args.mode}")
    print(f"include_short:  {args.include_short}")
    print(f"balanced_train: {args.balanced_train}")
    print(f"max_protein_len:{args.max_protein_len}")
    print(f"hidden_dim:     {args.hidden_dim}")
    print(f"protein_proj_h: {args.protein_projection_hidden_dim}")
    print(f"protein_comp:   {args.protein_compression}")
    print(f"protein_latent: {args.protein_latent_len}")
    print(f"binary_pooling: {args.binary_pooling}")
    print(f"binary_alpha:   {args.binary_alpha_source}")
    if args.balanced_train:
        print(f"balanced_pos:   {args.balanced_pos_fraction}")
        print(f"steps/epoch:    {args.steps_per_epoch or 1000}")
    print(f"tracks:         {'all matched tracks' if track_indices is None else track_indices}")
    print(f"cell_vocab:     {cell_to_index}")
    print(f"train samples:  {len(train_dataset)}")
    print(f"valid samples:  {len(valid_dataset)}")
    if args.valid_sample_size is not None:
        print(f"valid sample:   {min(args.valid_sample_size, len(valid_dataset))} fixed random flattened samples")
    print("loading frozen PARNET...", flush=True)
    parnet = load_parnet(device=device)

    probe = move_batch(next(iter(train_loader)), device)
    with torch.no_grad():
        probe_features = parnet.body_feats(probe["onehot"])
    head = ProteinCellCrossAttentionProfileHead(
        protein_dim=int(probe["protein_residue_embedding"].shape[-1]),
        rna_channels=int(probe_features.shape[1]),
        cell_count=len(cell_to_index),
        cell_dim=args.cell_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        protein_projection_hidden_dim=args.protein_projection_hidden_dim,
        protein_compression=args.protein_compression,
        protein_latent_len=args.protein_latent_len,
        binary_pooling=args.binary_pooling,
        binary_alpha_source=args.binary_alpha_source,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume is not None and args.run_name is None:
        run_name = args.resume.parent.name
    else:
        run_name = args.run_name or f"cross_attention_{args.mode}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_config = {
        **vars(args),
        "hfds": str(args.hfds),
        "binding_dataset": str(args.binding_dataset),
        "track_map": str(args.track_map),
        "protein_h5": str(args.protein_h5),
        "out_dir": str(out_dir),
        "device": device,
        "cell_to_index": cell_to_index,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
    }
    metrics = {"config": metrics_config, "epochs": []}

    best_pearson = float("-inf")
    best_auprc = float("-inf")
    completed_epochs = 0
    if args.resume is not None:
        print(f"resuming from:  {args.resume}", flush=True)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        head.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_epochs = int(checkpoint.get("epoch", 0))
        best_pearson = float(checkpoint.get("best_pearson", float("-inf")))
        best_auprc = float(checkpoint.get("best_auprc", float("-inf")))
        metrics = checkpoint.get("metrics", metrics)
        if best_auprc == float("-inf") and metrics.get("epochs"):
            best_auprc = max(row["valid"].get("binding_auprc", float("-inf")) for row in metrics["epochs"])
        metrics.setdefault("resume_history", []).append(
            {
                "checkpoint": str(args.resume),
                "completed_epochs": completed_epochs,
                "new_args": metrics_config,
            }
        )
        if hasattr(train_loader.sampler, "epoch"):
            train_loader.sampler.epoch = completed_epochs

    final_epoch = completed_epochs + args.epochs
    for epoch in range(completed_epochs + 1, final_epoch + 1):
        print(f"\nepoch {epoch}/{final_epoch}", flush=True)
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
            lambda_profile=args.lambda_profile,
            lambda_binary=args.lambda_binary,
            binary_pos_weight=args.binary_pos_weight,
            profile_mask_source=args.profile_mask_source,
            task=args.task,
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
                lambda_profile=args.lambda_profile,
                lambda_binary=args.lambda_binary,
                binary_pos_weight=args.binary_pos_weight,
                profile_mask_source=args.profile_mask_source,
                task=args.task,
                progress_every=0,
            )
        row = {"epoch": epoch, "train": train_stats, "valid": valid_stats}
        metrics["epochs"].append(row)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        print(
            f"  train loss={train_stats['loss']:.4f} profile={train_stats['profile_loss']:.4f} "
            f"binary={train_stats['binary_loss']:.4f} pearson={train_stats['pearson']:+.4f} "
            f"n={train_stats['n_profiles']} bind_pos={train_stats['binding_pos_rate']:.4f} "
            f"bind_acc={train_stats['binding_accuracy']:.3f} bind_auprc={train_stats['binding_auprc']:.4f}"
        )
        print(
            f"  valid loss={valid_stats['loss']:.4f} profile={valid_stats['profile_loss']:.4f} "
            f"binary={valid_stats['binary_loss']:.4f} pearson={valid_stats['pearson']:+.4f} "
            f"n={valid_stats['n_profiles']} bind_pos={valid_stats['binding_pos_rate']:.4f} "
            f"bind_acc={valid_stats['binding_accuracy']:.3f} bind_auprc={valid_stats['binding_auprc']:.4f}"
        )

        checkpoint_kwargs = {
            "head": head,
            "optimizer": optimizer,
            "cell_to_index": cell_to_index,
            "protein_dim": int(probe["protein_residue_embedding"].shape[-1]),
            "rna_channels": int(probe_features.shape[1]),
            "args": args,
            "epoch": epoch,
            "metrics": metrics,
            "valid_stats": valid_stats,
        }
        if valid_stats["pearson"] > best_pearson:
            best_pearson = valid_stats["pearson"]
            save_training_checkpoint(
                out_dir / "best.pt",
                best_pearson=best_pearson,
                best_auprc=best_auprc,
                **checkpoint_kwargs,
            )
            print(f"  saved new best checkpoint: {out_dir / 'best.pt'}", flush=True)
            save_training_checkpoint(
                out_dir / "best_pearson.pt",
                best_pearson=best_pearson,
                best_auprc=best_auprc,
                **checkpoint_kwargs,
            )
            print(f"  saved new best Pearson checkpoint: {out_dir / 'best_pearson.pt'}", flush=True)
        if valid_stats["binding_auprc"] > best_auprc:
            best_auprc = valid_stats["binding_auprc"]
            save_training_checkpoint(
                out_dir / "best_auprc.pt",
                best_pearson=best_pearson,
                best_auprc=best_auprc,
                **checkpoint_kwargs,
            )
            print(f"  saved new best AUPRC checkpoint: {out_dir / 'best_auprc.pt'}", flush=True)
        save_training_checkpoint(
            out_dir / "last.pt",
            best_pearson=best_pearson,
            best_auprc=best_auprc,
            **checkpoint_kwargs,
        )

    torch.save(head.state_dict(), out_dir / "last.statedict.pt")
    print(f"\nwrote metrics: {out_dir / 'metrics.json'}")
    print(f"wrote last:    {out_dir / 'last.pt'}")
    print(f"wrote weights: {out_dir / 'last.statedict.pt'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mmpartnet_pycache")
    main()
