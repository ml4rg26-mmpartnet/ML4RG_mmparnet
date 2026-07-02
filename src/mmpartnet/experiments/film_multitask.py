#!/usr/bin/env python
"""Train and evaluate the protein+cell FiLM multitask baseline.

This module contains the reusable experiment logic. The ``scripts/`` entry
points are intentionally thin wrappers around ``train_main`` and ``eval_main``
so the workflow follows the main branch's package layout.

The model is the minimal end-to-end multimodal experiment:

    multimodal batch -> frozen PARNET body_feats -> ProteinCellFiLMProfileHead
      -> RBPNet-style profile loss against eCLIP/control counts
      -> binary binding loss against narrowPeak/pureCLIP labels

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
from torch.utils.data import DataLoader, Sampler

from mmpartnet.data.multimodal import (
    MultimodalCollator,
    ParnetMultimodalDataset,
    build_cell_vocab,
    load_track_protein_map,
)
from mmpartnet.models import EarlyFusionConcatHead, ProteinCellFiLMProfileHead, load_parnet


SHARED = Path("/home/dgu/storage_ml4rg26-shared")
MMPARNET = Path("/home/dgu/storage_ml4rg26-mmparnet")
REPO = Path(__file__).resolve().parents[3]
DEFAULT_HFDS = (
    SHARED
    / "parnet-eclip/data-formatted-for-training/"
    / "600nt_windows.no-one-hot.stripped/encode.filtered.hfds"
)
DEFAULT_TRACK_MAP = REPO / "mmpartnet_out/prott5_track_map.tsv"
DEFAULT_PROTEIN_H5 = MMPARNET / "manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5"
DEFAULT_BINDING = (
    MMPARNET
    / "manually_gathered/600nt_windows.no-one-hot.stripped.binding/"
    / "600nt_windows.no-one-hot.stripped.binding.narrowpeak_intersect/dataset.pt"
)
DEFAULT_OUT = REPO / "mmpartnet_out/film_runs"


class BindingBalancedSampler(Sampler[int]):
    """Sample window-track pairs with a requested positive-label fraction.

    The underlying dataset is flattened as:

        dataset_index = window_offset * n_tracks + track_offset

    Positive pairs are rare in the binary binding dataset, so plain shuffle often
    creates batches with no positive examples. This sampler keeps all tensors and
    labels unchanged; it only changes which flattened pair indices are visited.
    """

    def __init__(
        self,
        dataset: ParnetMultimodalDataset,
        *,
        positive_fraction: float,
        num_samples: int,
        seed: int,
    ):
        if dataset.binding_split is None:
            raise ValueError("balanced sampling requires binding labels")
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be between 0 and 1 inclusive")
        self.dataset = dataset
        self.positive_fraction = positive_fraction
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self.n_tracks = len(dataset.track_indices)
        self.track_indices_tensor = torch.tensor(dataset.track_indices, dtype=torch.long)
        self.positive_indices = self._find_positive_indices()
        if not self.positive_indices:
            raise ValueError("no positive binding labels found for this split/track selection")

    def _label_for_index(self, idx: int) -> float:
        window_offset = idx // self.n_tracks
        track_offset = idx % self.n_tracks
        window_index = self.dataset.window_indices[window_offset]
        track_index = self.dataset.track_indices[track_offset]
        binding = self.dataset.binding_split[window_index]["outputs"]["binding"]
        return float(binding[track_index])

    def _find_positive_indices(self) -> list[int]:
        positives = []
        for window_offset, window_index in enumerate(self.dataset.window_indices):
            binding = self.dataset.binding_split[window_index]["outputs"]["binding"]
            base = window_offset * self.n_tracks
            selected = binding[self.track_indices_tensor]
            positive_offsets = torch.nonzero(selected > 0.5, as_tuple=False).flatten().tolist()
            positives.extend(base + int(track_offset) for track_offset in positive_offsets)
        return positives

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        total = len(self.dataset)
        n_pos = len(self.positive_indices)
        for _ in range(self.num_samples):
            draw_positive = torch.rand((), generator=generator).item() < self.positive_fraction
            if draw_positive:
                pos_i = int(torch.randint(n_pos, (1,), generator=generator).item())
                yield self.positive_indices[pos_i]
                continue
            while True:
                idx = int(torch.randint(total, (1,), generator=generator).item())
                if self._label_for_index(idx) <= 0.5:
                    yield idx
                    break

    def __len__(self) -> int:
        return self.num_samples


def parse_tracks(value: str) -> list[int] | None:
    if value.lower() == "all":
        return None
    return [int(x) for x in value.split(",") if x.strip()]


def make_loader(
    hfds,
    binding_data,
    split: str,
    track_map,
    track_indices: list[int] | None,
    protein_h5: Path,
    protein_rep: str,
    track_map_path: Path,
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
    collator = MultimodalCollator(
        protein_h5,
        seq_len=seq_len,
        cell_to_index=cell_to_index,
        protein_rep=protein_rep,
        track_map=track_map_path,
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
    for key in ("onehot", "mask", "protein_embedding", "cell_index", "eclip", "control"):
        out[key] = batch[key].to(device)
    if "binding" in batch:
        out["binding"] = batch["binding"].to(device)
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
    valid_mask: torch.Tensor | None = None,
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
    if valid_mask is not None:
        valid = valid & valid_mask.bool()
    corr = numerator / denominator.clamp_min(1e-9)
    return torch.where(valid, corr, torch.zeros_like(corr)).sum(), valid.sum()


def binary_stats(logit: torch.Tensor, label: torch.Tensor) -> dict:
    prob = torch.sigmoid(logit.detach())
    label = label.detach().float()
    pred = prob >= 0.5
    correct = (pred == (label >= 0.5)).sum()
    pos = label.sum()
    pred_pos = pred.sum()
    tp = ((pred == 1) & (label == 1)).sum()
    precision = tp / pred_pos.clamp_min(1)
    recall = tp / pos.clamp_min(1)
    return {
        "n": int(label.numel()),
        "pos": int(pos.detach().cpu()),
        "accuracy_sum": int(correct.detach().cpu()),
        "precision_sum": float(precision.detach().cpu()),
        "recall_sum": float(recall.detach().cpu()),
    }


def binary_average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average precision / AUPRC for binary labels.

    Accuracy is misleading when positives are rare. Average precision asks:
    among the samples the model ranks highly, how many are true positives?
    """
    labels = labels.float()
    positives = labels.sum()
    if int(positives.item()) == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    ranked_labels = labels[order]
    true_positives = torch.cumsum(ranked_labels, dim=0)
    ranks = torch.arange(1, ranked_labels.numel() + 1, dtype=torch.float32)
    precision_at_k = true_positives / ranks
    ap = (precision_at_k * ranked_labels).sum() / positives
    return float(ap.item())


def save_training_checkpoint(
    path: Path,
    *,
    head: ProteinCellFiLMProfileHead,
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
    head: ProteinCellFiLMProfileHead,
    loader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    task: str,
    mode: str,
    min_count: float,
    max_batches: int | None,
    mix_penalty: float,
    lambda_profile: float,
    lambda_binary: float,
    binary_pos_weight: float | None,
    profile_mask_source: str,
    progress_every: int,
) -> dict:
    training = optimizer is not None
    head.train(training)
    loss_sum = 0.0
    loss_n = 0
    pear_sum = torch.tensor(0.0, device=device)
    pear_n = torch.tensor(0, device=device)
    all_pear_sum = torch.tensor(0.0, device=device)
    all_pear_n = torch.tensor(0, device=device)
    profile_loss_sum = 0.0
    binary_loss_sum = 0.0
    profile_mask_n = 0
    binding_n = 0
    binding_pos = 0
    binding_correct = 0
    binary_scores = []
    binary_labels = []

    for step, raw_batch in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        batch = move_batch(raw_batch, device)
        protein, cell_index = apply_mode(batch, mode)
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
            out = head(rna_features, protein, cell_index, mask=batch["mask"], task=task)
            if task in {"multitask", "profile-only"}:
                pred = out["total"]
                ps, pn = pearson_sum(pred, batch["eclip"], min_count, mask=batch["mask"], valid_mask=profile_mask)
                pear_sum += ps
                pear_n += pn
                all_ps, all_pn = pearson_sum(pred, batch["eclip"], min_count, mask=batch["mask"])
                all_pear_sum += all_ps
                all_pear_n += all_pn
            if task in {"multitask", "binary-only"} and binding is not None:
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
    mean_all_pearson = float((all_pear_sum / all_pear_n.clamp_min(1)).detach().cpu())
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
        "all_evaluable_pearson": mean_all_pearson,
        "all_evaluable_n_profiles": int(all_pear_n.detach().cpu()),
        "profile_mask_n": profile_mask_n,
        "binding_n": binding_n,
        "binding_pos": binding_pos,
        "binding_pos_rate": binding_pos / max(binding_n, 1),
        "binding_accuracy": binding_correct / max(binding_n, 1),
        "binding_auprc": binding_auprc,
    }


def train_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hfds", type=Path, default=DEFAULT_HFDS)
    parser.add_argument("--binding-dataset", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--track-map", type=Path, default=DEFAULT_TRACK_MAP)
    parser.add_argument(
        "--protein-rep",
        default="prott5_h5",
        help="Protein provider registry name from mmpartnet.protein. Default: prott5_h5.",
    )
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
    parser.add_argument("--balanced-train", action="store_true")
    parser.add_argument("--balanced-pos-fraction", type=float, default=0.5)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Number of training batches per epoch when --balanced-train is used. Default: 1000.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-count", type=float, default=10.0)
    parser.add_argument("--mix-penalty", type=float, default=0.0)
    parser.add_argument("--task", default="multitask", choices=["multitask", "binary-only", "profile-only"])
    parser.add_argument("--arch", default="film", choices=["film", "concat"],
                    help="Head architecture: film = ProteinCellFiLMProfileHead, "
                         "concat = EarlyFusionConcatHead (binary-only baseline).")
    parser.add_argument("--lambda-profile", type=float, default=1.0)
    parser.add_argument("--lambda-binary", type=float, default=1.0)
    parser.add_argument("--binary-pos-weight", type=float, default=None)
    parser.add_argument(
        "--profile-mask-source",
        default="binding",
        choices=["binding", "count", "binding-and-count"],
        help="Which true labels decide where profile loss/Pearson are computed.",
    )
    parser.add_argument("--mode", default="multimodal", choices=["multimodal", "rna-only", "protein-shuffle"])
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume head/optimizer/metrics from a full checkpoint, usually mmpartnet_out/film_runs/<run>/last.pt.",
    )
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
        args.protein_rep,
        args.track_map,
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
    )
    valid_dataset, valid_loader = make_loader(
        hfds,
        binding_data,
        args.valid_split,
        track_map,
        track_indices,
        args.protein_h5,
        args.protein_rep,
        args.track_map,
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
    )

    print(f"device:         {device}")
    print(f"task:           {args.task}")
    print(f"mode:           {args.mode}")
    print(f"include_short:  {args.include_short}")
    print(f"balanced_train: {args.balanced_train}")
    if args.balanced_train:
        print(f"balanced_pos:   {args.balanced_pos_fraction}")
        print(f"steps/epoch:    {args.steps_per_epoch or 1000}")
    print(f"tracks:         {'all matched tracks' if track_indices is None else track_indices}")
    print(f"protein_rep:    {args.protein_rep}")
    print(f"cell_vocab:     {cell_to_index}")
    print(f"train samples:  {len(train_dataset)}")
    print(f"valid samples:  {len(valid_dataset)}")
    print("loading frozen PARNET...", flush=True)
    parnet = load_parnet(device=device)

    probe = move_batch(next(iter(train_loader)), device)
    with torch.no_grad():
        probe_features = parnet.body_feats(probe["onehot"])
    
    head_cls = EarlyFusionConcatHead if args.arch == "concat" else ProteinCellFiLMProfileHead
    head = head_cls(
        protein_dim=int(probe["protein_embedding"].shape[1]),
        rna_channels=int(probe_features.shape[1]),
        cell_count=len(cell_to_index),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume is not None and args.run_name is None:
        run_name = args.resume.parent.name
    else:
        run_name = args.run_name or f"film_{args.mode}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_config = {
        **vars(args),
        "hfds": str(args.hfds),
        "binding_dataset": str(args.binding_dataset),
        "track_map": str(args.track_map),
        "protein_rep": args.protein_rep,
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
            task=args.task,
            mode=args.mode,
            min_count=args.min_count,
            max_batches=args.max_train_batches,
            mix_penalty=args.mix_penalty,
            lambda_profile=args.lambda_profile,
            lambda_binary=args.lambda_binary,
            binary_pos_weight=args.binary_pos_weight,
            profile_mask_source=args.profile_mask_source,
            progress_every=args.progress_every,
        )
        with torch.no_grad():
            valid_stats = run_epoch(
                parnet=parnet,
                head=head,
                loader=valid_loader,
                optimizer=None,
                device=device,
                task=args.task,
                mode=args.mode,
                min_count=args.min_count,
                max_batches=args.max_valid_batches,
                mix_penalty=args.mix_penalty,
                lambda_profile=args.lambda_profile,
                lambda_binary=args.lambda_binary,
                binary_pos_weight=args.binary_pos_weight,
                profile_mask_source=args.profile_mask_source,
                progress_every=0,
            )
        row = {"epoch": epoch, "train": train_stats, "valid": valid_stats}
        metrics["epochs"].append(row)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        print(
            f"  train loss={train_stats['loss']:.4f} profile={train_stats['profile_loss']:.4f} "
            f"binary={train_stats['binary_loss']:.4f} pearson={train_stats['pearson']:+.4f} "
            f"n={train_stats['n_profiles']} all_pearson={train_stats['all_evaluable_pearson']:+.4f} "
            f"all_n={train_stats['all_evaluable_n_profiles']} bind_pos={train_stats['binding_pos_rate']:.4f} "
            f"bind_acc={train_stats['binding_accuracy']:.3f} bind_auprc={train_stats['binding_auprc']:.4f}"
        )
        print(
            f"  valid loss={valid_stats['loss']:.4f} profile={valid_stats['profile_loss']:.4f} "
            f"binary={valid_stats['binary_loss']:.4f} pearson={valid_stats['pearson']:+.4f} "
            f"n={valid_stats['n_profiles']} all_pearson={valid_stats['all_evaluable_pearson']:+.4f} "
            f"all_n={valid_stats['all_evaluable_n_profiles']} bind_pos={valid_stats['binding_pos_rate']:.4f} "
            f"bind_acc={valid_stats['binding_accuracy']:.3f} bind_auprc={valid_stats['binding_auprc']:.4f}"
        )

        if valid_stats["pearson"] > best_pearson:
            best_pearson = valid_stats["pearson"]
            save_training_checkpoint(
                out_dir / "best.pt",
                head=head,
                optimizer=optimizer,
                cell_to_index=cell_to_index,
                protein_dim=int(probe["protein_embedding"].shape[1]),
                rna_channels=int(probe_features.shape[1]),
                args=args,
                epoch=epoch,
                best_pearson=best_pearson,
                best_auprc=best_auprc,
                metrics=metrics,
                valid_stats=valid_stats,
            )
            print(f"  saved new best checkpoint: {out_dir / 'best.pt'}", flush=True)
            save_training_checkpoint(
                out_dir / "best_pearson.pt",
                head=head,
                optimizer=optimizer,
                cell_to_index=cell_to_index,
                protein_dim=int(probe["protein_embedding"].shape[1]),
                rna_channels=int(probe_features.shape[1]),
                args=args,
                epoch=epoch,
                best_pearson=best_pearson,
                best_auprc=best_auprc,
                metrics=metrics,
                valid_stats=valid_stats,
            )
            print(f"  saved new best Pearson checkpoint: {out_dir / 'best_pearson.pt'}", flush=True)
        if valid_stats["binding_auprc"] > best_auprc:
            best_auprc = valid_stats["binding_auprc"]
            save_training_checkpoint(
                out_dir / "best_auprc.pt",
                head=head,
                optimizer=optimizer,
                cell_to_index=cell_to_index,
                protein_dim=int(probe["protein_embedding"].shape[1]),
                rna_channels=int(probe_features.shape[1]),
                args=args,
                epoch=epoch,
                best_pearson=best_pearson,
                best_auprc=best_auprc,
                metrics=metrics,
                valid_stats=valid_stats,
            )
            print(f"  saved new best AUPRC checkpoint: {out_dir / 'best_auprc.pt'}", flush=True)
        save_training_checkpoint(
            out_dir / "last.pt",
            head=head,
            optimizer=optimizer,
            cell_to_index=cell_to_index,
            protein_dim=int(probe["protein_embedding"].shape[1]),
            rna_channels=int(probe_features.shape[1]),
            args=args,
            epoch=epoch,
            best_pearson=best_pearson,
            best_auprc=best_auprc,
            metrics=metrics,
            valid_stats=valid_stats,
        )

    torch.save(head.state_dict(), out_dir / "last.statedict.pt")
    print(f"\nwrote metrics: {out_dir / 'metrics.json'}")
    print(f"wrote last:    {out_dir / 'last.pt'}")
    print(f"wrote weights: {out_dir / 'last.statedict.pt'}")


def checkpoint_args(checkpoint: dict) -> dict:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def arg_or_checkpoint(cli_value, ckpt_args: dict, name: str, default=None):
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(name, default)


def eval_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained protein+cell FiLM multitask checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full checkpoint, usually best.pt or last.pt.")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--arch", default="film", choices=["film", "concat"], help="Head architecture (must match how the checkpoint was trained).")
    parser.add_argument("--hfds", type=Path, default=None)
    parser.add_argument("--binding-dataset", type=Path, default=None)
    parser.add_argument("--track-map", type=Path, default=None)
    parser.add_argument("--protein-rep", default=None)
    parser.add_argument("--protein-h5", type=Path, default=None)
    parser.add_argument("--tracks", default=None, help="Comma-separated track indices, 'all', or checkpoint value.")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--include-short", action="store_true", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=[None, "multimodal", "rna-only", "protein-shuffle"])
    parser.add_argument("--min-count", type=float, default=None)
    parser.add_argument("--mix-penalty", type=float, default=None)
    parser.add_argument("--task", default=None, choices=[None, "multitask", "binary-only", "profile-only"])
    parser.add_argument("--lambda-profile", type=float, default=None)
    parser.add_argument("--lambda-binary", type=float, default=None)
    parser.add_argument("--binary-pos-weight", type=float, default=None)
    parser.add_argument(
        "--profile-mask-source",
        default=None,
        choices=[None, "binding", "count", "binding-and-count"],
    )
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
    protein_rep = arg_or_checkpoint(args.protein_rep, ckpt_args, "protein_rep", "prott5_h5")
    tracks_value = arg_or_checkpoint(args.tracks, ckpt_args, "tracks", "all")
    max_windows_raw = arg_or_checkpoint(args.max_windows, ckpt_args, f"max_{args.split}_windows", 0)
    max_windows = None if int(max_windows_raw) == 0 else int(max_windows_raw)
    seq_len = int(arg_or_checkpoint(args.seq_len, ckpt_args, "seq_len", 600))
    include_short = bool(arg_or_checkpoint(args.include_short, ckpt_args, "include_short", False))
    batch_size = int(arg_or_checkpoint(args.batch_size, ckpt_args, "batch_size", 32))
    task = arg_or_checkpoint(args.task, ckpt_args, "task", "multitask")
    mode = arg_or_checkpoint(args.mode, ckpt_args, "mode", "multimodal")
    min_count = float(arg_or_checkpoint(args.min_count, ckpt_args, "min_count", 10.0))
    mix_penalty = float(arg_or_checkpoint(args.mix_penalty, ckpt_args, "mix_penalty", 0.0))
    lambda_profile = float(arg_or_checkpoint(args.lambda_profile, ckpt_args, "lambda_profile", 1.0))
    lambda_binary = float(arg_or_checkpoint(args.lambda_binary, ckpt_args, "lambda_binary", 1.0))
    binary_pos_weight = arg_or_checkpoint(args.binary_pos_weight, ckpt_args, "binary_pos_weight", None)
    profile_mask_source = arg_or_checkpoint(args.profile_mask_source, ckpt_args, "profile_mask_source", "binding")

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
        str(protein_rep),
        track_map_path,
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
    )

    print(f"checkpoint:     {args.checkpoint}")
    print(f"device:         {device}")
    print(f"split:          {args.split}")
    print(f"task:           {task}")
    print(f"mode:           {mode}")
    print(f"include_short:  {include_short}")
    print(f"tracks:         {'all matched tracks' if track_indices is None else track_indices}")
    print(f"protein_rep:    {protein_rep}")
    print(f"samples:        {len(dataset)}")
    print(f"max_batches:    {args.max_batches}")
    print("loading frozen PARNET...", flush=True)
    parnet = load_parnet(device=device)

    head_cls = EarlyFusionConcatHead if args.arch == "concat" else ProteinCellFiLMProfileHead
    head = head_cls(
        protein_dim=int(checkpoint["protein_dim"]),
        rna_channels=int(checkpoint["rna_channels"]),
        cell_count=len(cell_to_index),
    ).to(device)
    head.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        stats = run_epoch(
            parnet=parnet,
            head=head,
            loader=loader,
            optimizer=None,
            device=device,
            task=task,
            mode=mode,
            min_count=min_count,
            max_batches=args.max_batches,
            mix_penalty=mix_penalty,
            lambda_profile=lambda_profile,
            lambda_binary=lambda_binary,
            binary_pos_weight=binary_pos_weight,
            profile_mask_source=profile_mask_source,
            progress_every=0,
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "config": {
            "hfds": str(hfds_path),
            "binding_dataset": str(binding_path),
            "track_map": str(track_map_path),
            "protein_rep": protein_rep,
            "protein_h5": str(protein_h5),
            "tracks": tracks_value,
            "max_windows": max_windows,
            "batch_size": batch_size,
            "max_batches": args.max_batches,
            "include_short": include_short,
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
    # Keep PARNET import stubs and torch caches out of the repo when the script is run from a shared mount.
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mmpartnet_pycache")
    train_main()
