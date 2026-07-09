#!/usr/bin/env python
"""Evaluate one cross-attention checkpoint per RBP-cell track."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for path in (SRC, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from mmpartnet.data.multimodal import (
    MultimodalCollator,
    ParnetMultimodalDataset,
    build_cell_vocab,
    load_track_protein_map,
)
from mmpartnet.models import load_parnet
from mmpartnet.models.cross_attention_dgu import ProteinCellCrossAttentionProfileHead
from scripts.train_cross_attention_profile import (
    DEFAULT_BINDING,
    DEFAULT_HFDS,
    DEFAULT_PROTEIN_H5,
    DEFAULT_TRACK_MAP,
    apply_mode,
    move_batch,
)
from scripts.train_film_profile import binary_average_precision


def checkpoint_args(checkpoint: dict) -> dict:
    value = checkpoint.get("args", {})
    return value if isinstance(value, dict) else {}


def resolve_path(cli_value: Path | None, saved: dict, name: str, default: Path) -> Path:
    return Path(cli_value if cli_value is not None else saved.get(name, default))


def load_common68_track_indices(binding_fair: Path, track_map) -> list[int]:
    """Recover the ordered 68-track comparison panel from binding_fair.json."""
    data = json.loads(binding_fair.read_text(encoding="utf-8"))
    rbps = data.get("rbps")
    if data.get("K") != 68 or not isinstance(rbps, list):
        raise ValueError(f"{binding_fair} does not describe the expected K=68 panel")
    rbp_set = set(rbps)
    selected = [row for row in track_map if row.rbp in rbp_set]
    if [row.rbp for row in selected] != rbps:
        raise ValueError("Could not reproduce the ordered 68-track panel from the track map")
    return [row.track_index for row in selected]


def choose_windows(dataset: ParnetMultimodalDataset, selection: str, count: int, seed: int) -> list[int]:
    eligible = dataset.window_indices
    keep = min(count, len(eligible))
    if selection == "reference-first":
        return eligible[:keep]
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randperm(len(eligible), generator=generator)[:keep].tolist()
    return [eligible[i] for i in offsets]


def profile_correlations(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    binding: torch.Tensor | None,
    min_count: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.float()
    valid_len = mask_f.sum(dim=-1).clamp_min(1)
    pred_mean = (pred * mask_f).sum(dim=-1) / valid_len
    true_mean = (true * mask_f).sum(dim=-1) / valid_len
    pred_centered = (pred - pred_mean[:, None]) * mask_f
    true_centered = (true - true_mean[:, None]) * mask_f
    numerator = (pred_centered * true_centered).sum(dim=-1)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=-1) * true_centered.square().sum(dim=-1)
    )
    valid = (true.sum(dim=-1) >= min_count) & (denominator > 1e-9)
    if binding is not None:
        valid &= binding > 0.5
    corr = numerator / denominator.clamp_min(1e-9)
    return corr, valid


def make_head(checkpoint: dict, cell_count: int, device: str) -> ProteinCellCrossAttentionProfileHead:
    config = checkpoint.get("model_config", {})
    head = ProteinCellCrossAttentionProfileHead(
        protein_dim=int(checkpoint["protein_dim"]),
        rna_channels=int(checkpoint["rna_channels"]),
        cell_count=cell_count,
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
    head.eval()
    return head


def optional_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=["profile-only", "binary-only", "multitask"])
    parser.add_argument("--panel", required=True, choices=["common68", "all"])
    parser.add_argument("--window-selection", required=True, choices=["reference-first", "random"])
    parser.add_argument("--window-count", type=int, default=15000)
    parser.add_argument("--window-seed", type=int, default=2026)
    parser.add_argument("--hfds", type=Path)
    parser.add_argument("--binding-dataset", type=Path)
    parser.add_argument("--track-map", type=Path)
    parser.add_argument("--protein-h5", type=Path)
    parser.add_argument("--binding-fair", type=Path, default=REPO / "mmpartnet_out/binding_fair.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-count", type=float, default=10.0)
    parser.add_argument("--max-protein-len", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = checkpoint_args(checkpoint)
    hfds_path = resolve_path(args.hfds, saved, "hfds", DEFAULT_HFDS)
    binding_path = resolve_path(args.binding_dataset, saved, "binding_dataset", DEFAULT_BINDING)
    track_map_path = resolve_path(args.track_map, saved, "track_map", DEFAULT_TRACK_MAP)
    protein_h5 = resolve_path(args.protein_h5, saved, "protein_h5", DEFAULT_PROTEIN_H5)
    seq_len = int(saved.get("seq_len", 600))
    max_protein_len = args.max_protein_len
    if max_protein_len is None and saved.get("max_protein_len") is not None:
        max_protein_len = int(saved["max_protein_len"])

    track_map = load_track_protein_map(track_map_path)
    track_indices = (
        load_common68_track_indices(args.binding_fair, track_map)
        if args.panel == "common68"
        else [row.track_index for row in track_map]
    )
    cell_to_index = checkpoint.get("cell_to_index") or build_cell_vocab(track_map)
    hfds = load_from_disk(str(hfds_path))
    binding_data = torch.load(binding_path, map_location="cpu", weights_only=False)

    # The comparison panel uses the first 15k exactly-600-nt test windows. The expanded panel
    # uses a saved deterministic random sample from every test window <=600 nt.
    exact_length = seq_len if args.window_selection == "reference-first" else None
    dataset = ParnetMultimodalDataset(
        hfds["test"],
        track_map,
        binding_split=binding_data["test"],
        track_indices=track_indices,
        exact_length=exact_length,
        max_length=None if exact_length is not None else seq_len,
    )
    dataset.window_indices = choose_windows(
        dataset, args.window_selection, args.window_count, args.window_seed
    )
    collator = MultimodalCollator(
        protein_h5,
        seq_len=seq_len,
        cell_to_index=cell_to_index,
        return_residue_embeddings=True,
        max_protein_len=max_protein_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    print(
        f"task={args.task} panel={args.panel} tracks={len(track_indices)} "
        f"windows={len(dataset.window_indices)} samples={len(dataset)} device={device}",
        flush=True,
    )
    print("loading frozen PARNET...", flush=True)
    parnet = load_parnet(device=device)
    head = make_head(checkpoint, len(cell_to_index), device)
    mode = str(saved.get("mode", "multimodal"))

    pearson_sum: dict[int, float] = defaultdict(float)
    pearson_n: dict[int, int] = defaultdict(int)
    binary_scores: dict[int, list[torch.Tensor]] = defaultdict(list)
    binary_labels: dict[int, list[torch.Tensor]] = defaultdict(list)
    sample_n: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for step, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, device)
            protein, protein_mask, cell_index = apply_mode(batch, mode)
            # Flattened batches repeat each RNA window once per track. Run the
            # frozen PARNET body once per unique window and expand its features.
            _, inverse = torch.unique(raw_batch["window_index"], sorted=True, return_inverse=True)
            n_unique = int(inverse.max()) + 1
            first = torch.stack(
                [(inverse == unique_index).nonzero(as_tuple=False)[0, 0] for unique_index in range(n_unique)]
            ).to(device)
            unique_features = parnet.body_feats(batch["onehot"][first]).detach()
            rna_features = unique_features[inverse.to(device)]
            out = head(
                rna_features,
                protein,
                cell_index,
                mask=batch["mask"],
                protein_mask=protein_mask,
                task=args.task,
            )
            track_batch = raw_batch["track_index"]
            binding = batch.get("binding")

            correlations = valid_profile = None
            if "total" in out:
                correlations, valid_profile = profile_correlations(
                    out["total"], batch["eclip"], batch["mask"], binding, args.min_count
                )
                correlations = correlations.cpu()
                valid_profile = valid_profile.cpu()
            scores = (
                torch.sigmoid(out["binding_logit"]).cpu()
                if "binding_logit" in out and binding is not None
                else None
            )
            labels = binding.cpu() if binding is not None else None

            for track_index in track_batch.unique().tolist():
                selected = track_batch == track_index
                sample_n[track_index] += int(selected.sum())
                if correlations is not None and valid_profile is not None:
                    keep = selected & valid_profile
                    if bool(keep.any()):
                        pearson_sum[track_index] += float(correlations[keep].sum())
                        pearson_n[track_index] += int(keep.sum())
                if scores is not None and labels is not None:
                    binary_scores[track_index].append(scores[selected])
                    binary_labels[track_index].append(labels[selected])

            if args.progress_every and step % args.progress_every == 0:
                print(f"  batches={step}/{len(loader)}", flush=True)

    by_index = {row.track_index: row for row in track_map}
    rows = []
    for track_index in track_indices:
        metadata = by_index[track_index]
        labels = (
            torch.cat(binary_labels[track_index])
            if binary_labels.get(track_index)
            else torch.empty(0)
        )
        scores = (
            torch.cat(binary_scores[track_index])
            if binary_scores.get(track_index)
            else torch.empty(0)
        )
        n_positive = int(labels.sum()) if labels.numel() else None
        auprc = (
            binary_average_precision(scores, labels)
            if labels.numel() and n_positive and n_positive >= 5
            else None
        )
        rows.append(
            {
                "track_index": track_index,
                "rbp": metadata.rbp,
                "cell": metadata.cell,
                "rbp_cell": metadata.rbp_ct,
                "n_windows": sample_n[track_index],
                "n_positive": n_positive,
                "positive_rate": (
                    n_positive / labels.numel() if labels.numel() and n_positive is not None else None
                ),
                "n_profiles": pearson_n[track_index],
                "pearson": (
                    pearson_sum[track_index] / pearson_n[track_index]
                    if pearson_n[track_index]
                    else None
                ),
                "auprc": auprc,
            }
        )

    result = {
        "model": "ProteinCellCrossAttentionProfileHead",
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "panel": args.panel,
        "window_selection": args.window_selection,
        "window_seed": args.window_seed if args.window_selection == "random" else None,
        "window_indices": dataset.window_indices,
        "n_windows": len(dataset.window_indices),
        "n_tracks": len(track_indices),
        "track_indices": track_indices,
        "mean_pearson": optional_mean([row["pearson"] for row in rows]),
        "mean_auprc": optional_mean([row["auprc"] for row in rows]),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.out.with_suffix(".tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    pearson_text = "n/a" if result["mean_pearson"] is None else f"{result['mean_pearson']:.4f}"
    auprc_text = "n/a" if result["mean_auprc"] is None else f"{result['mean_auprc']:.4f}"
    print(f"wrote {args.out} | mean Pearson={pearson_text} mean AUPRC={auprc_text}", flush=True)


if __name__ == "__main__":
    main()
