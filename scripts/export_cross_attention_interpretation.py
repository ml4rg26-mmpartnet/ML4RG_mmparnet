#!/usr/bin/env python
"""Export per-sample cross-attention interpretation tensors.

This script saves a small validation/test subset with available length-600
distributions: target profile, binary-position distribution, final alpha_bind,
predicted profiles, true eCLIP/control counts, and optional motif-overlap
metrics.

Optional motif TSV format:

    rbp<TAB>motif
    QKI<TAB>ACUAAY

Motifs are matched on RNA sequences; DNA T is treated as U and common IUPAC
ambiguity codes are supported.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
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
from scripts.eval_cross_attention_profile import arg_or_checkpoint, checkpoint_args
from scripts.train_cross_attention_profile import (
    DEFAULT_BINDING,
    DEFAULT_HFDS,
    DEFAULT_PROTEIN_H5,
    DEFAULT_TRACK_MAP,
    apply_mode,
    make_loader,
    move_batch,
    parse_tracks,
)


IUPAC_RNA = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "T": "U",
    "R": "AG",
    "Y": "CU",
    "S": "GC",
    "W": "AU",
    "K": "GU",
    "M": "AC",
    "B": "CGU",
    "D": "AGU",
    "H": "ACU",
    "V": "ACG",
    "N": "ACGU",
}


def motif_to_regex(motif: str) -> re.Pattern:
    parts = []
    for ch in motif.upper().replace("T", "U"):
        if ch in IUPAC_RNA:
            chars = IUPAC_RNA[ch]
            parts.append(chars if len(chars) == 1 else f"[{chars}]")
        else:
            parts.append(re.escape(ch))
    return re.compile("".join(parts))


def load_motifs(path: Path | None) -> dict[str, list[tuple[str, re.Pattern]]]:
    if path is None:
        return {}
    motifs: dict[str, list[tuple[str, re.Pattern]]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "rbp" not in reader.fieldnames or "motif" not in reader.fieldnames:
            raise ValueError("motif TSV must have 'rbp' and 'motif' columns")
        for row in reader:
            rbp = row["rbp"].strip()
            motif = row["motif"].strip()
            if rbp and motif:
                motifs.setdefault(rbp, []).append((motif, motif_to_regex(motif)))
    return motifs


def motif_mask_for_sequence(seq: str, rbp: str, motifs: dict[str, list[tuple[str, re.Pattern]]], seq_len: int) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool)
    patterns = motifs.get(rbp, [])
    if not patterns:
        return mask
    rna = seq.upper().replace("T", "U")[:seq_len]
    for _motif, pattern in patterns:
        for match in pattern.finditer(rna):
            mask[match.start() : min(match.end(), seq_len)] = True
    return mask


def prob_on_mask(prob: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((prob * mask.to(dtype=prob.dtype)).sum().cpu())


def topk_overlap(prob: torch.Tensor, mask: torch.Tensor, k: int) -> float:
    if not bool(mask.any()):
        return 0.0
    k = min(k, prob.numel())
    top_idx = prob.topk(k).indices
    return float(mask[top_idx].to(dtype=torch.float32).mean().cpu())


def optional_batch_tensor(out: dict[str, torch.Tensor], key: str, index: int) -> torch.Tensor | None:
    value = out.get(key)
    if value is None:
        return None
    return value[index].cpu()


def optional_batch_float(out: dict[str, torch.Tensor], key: str, index: int) -> float | None:
    value = out.get(key)
    if value is None:
        return None
    return float(value[index].cpu())


def add_optional_motif_metric(
    row: dict[str, object],
    prob_key: str,
    topk_key: str,
    prob: torch.Tensor | None,
    motif_mask: torch.Tensor,
    topk: int,
) -> None:
    if prob is None:
        row[prob_key] = None
        row[topk_key] = None
        return
    row[prob_key] = prob_on_mask(prob, motif_mask)
    row[topk_key] = topk_overlap(prob, motif_mask, topk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--hfds", type=Path, default=None)
    parser.add_argument("--binding-dataset", type=Path, default=None)
    parser.add_argument("--track-map", type=Path, default=None)
    parser.add_argument("--protein-h5", type=Path, default=None)
    parser.add_argument("--tracks", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--include-short", action="store_true", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--task", default=None, choices=[None, "multitask", "profile-only", "binary-only"])
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-protein-len", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=[None, "multimodal", "rna-only", "protein-shuffle", "no-cell"])
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--motif-tsv", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True, help="Output .pt path with full interpretation records.")
    parser.add_argument("--motif-out", type=Path, default=None, help="Optional per-sample motif metrics TSV.")
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
    task = arg_or_checkpoint(args.task, ckpt_args, "task", "multitask")
    max_protein_len = arg_or_checkpoint(args.max_protein_len, ckpt_args, "max_protein_len", None)
    if max_protein_len is not None:
        max_protein_len = int(max_protein_len)
    mode = arg_or_checkpoint(args.mode, ckpt_args, "mode", "multimodal")

    track_indices = parse_tracks(str(tracks_value))
    track_map = load_track_protein_map(track_map_path)
    cell_to_index = checkpoint.get("cell_to_index") or build_cell_vocab(track_map)
    hfds = load_from_disk(str(hfds_path))
    binding_data = torch.load(binding_path, map_location="cpu", weights_only=False)
    _dataset, loader = make_loader(
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
    print(f"max_samples:    {args.max_samples}")
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
    head.eval()

    motifs = load_motifs(args.motif_tsv)
    records = []
    motif_rows = []
    with torch.no_grad():
        for step, raw_batch in enumerate(loader, start=1):
            if args.max_batches is not None and step > args.max_batches:
                break
            batch = move_batch(raw_batch, device)
            protein, protein_mask, cell_index = apply_mode(batch, mode)
            rna_features = parnet.body_feats(batch["onehot"]).detach()
            out = head(
                rna_features,
                protein,
                cell_index,
                mask=batch["mask"],
                protein_mask=protein_mask,
                task=task,
            )
            batch_size_actual = int(batch["onehot"].shape[0])
            for i in range(batch_size_actual):
                if len(records) >= args.max_samples:
                    break
                record = {
                    "window_index": int(raw_batch["window_index"][i]),
                    "window_name": raw_batch["window_name"][i],
                    "sequence": raw_batch["sequence"][i],
                    "track_index": int(raw_batch["track_index"][i]),
                    "rbp": raw_batch["rbp"][i],
                    "cell": raw_batch["cell"][i],
                    "rbp_ct": raw_batch["rbp_ct"][i],
                    "binding_label": float(raw_batch["binding"][i]) if "binding" in raw_batch else None,
                    "binding_logit": optional_batch_float(out, "binding_logit", i),
                    "binding_prob": optional_batch_float(out, "binding_prob", i),
                    "binding_gate": optional_batch_float(out, "binding_gate", i),
                    "target_prob": optional_batch_tensor(out, "target", i),
                    "binary_position_prob": optional_batch_tensor(out, "binary_position_prob", i),
                    "alpha_bind": optional_batch_tensor(out, "alpha_bind", i),
                    "pred_control": optional_batch_tensor(out, "control", i),
                    "pred_total": optional_batch_tensor(out, "total", i),
                    "mix_coeff": optional_batch_float(out, "mix_coeff", i),
                    "eclip_counts": raw_batch["eclip"][i].cpu(),
                    "control_counts": raw_batch["control"][i].cpu(),
                    "mask": raw_batch["mask"][i].cpu(),
                }
                if motifs:
                    motif_mask = motif_mask_for_sequence(record["sequence"], record["rbp"], motifs, seq_len)
                    record["motif_mask"] = motif_mask
                    row = {
                        "rbp": record["rbp"],
                        "cell": record["cell"],
                        "window_index": record["window_index"],
                        "track_index": record["track_index"],
                        "binding_label": record["binding_label"],
                        "binding_logit": record["binding_logit"],
                        "binding_prob": record["binding_prob"],
                        "binding_gate": record["binding_gate"],
                        "has_motif": int(bool(motif_mask.any())),
                    }
                    add_optional_motif_metric(
                        row,
                        "target_prob_on_motif",
                        "topk_target_overlap_motif",
                        record["target_prob"],
                        motif_mask,
                        args.topk,
                    )
                    add_optional_motif_metric(
                        row,
                        "binary_prob_on_motif",
                        "topk_binary_overlap_motif",
                        record["binary_position_prob"],
                        motif_mask,
                        args.topk,
                    )
                    add_optional_motif_metric(
                        row,
                        "alpha_bind_on_motif",
                        "topk_alpha_overlap_motif",
                        record["alpha_bind"],
                        motif_mask,
                        args.topk,
                    )
                    motif_rows.append(row)
                records.append(record)
            if len(records) >= args.max_samples:
                break

    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "task": task,
        "mode": mode,
        "seq_len": seq_len,
        "motif_tsv": None if args.motif_tsv is None else str(args.motif_tsv),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote interpretation records: {args.out}")

    if args.motif_out is not None and motif_rows:
        args.motif_out.parent.mkdir(parents=True, exist_ok=True)
        with args.motif_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(motif_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(motif_rows)
        print(f"wrote motif metrics:         {args.motif_out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mmpartnet_pycache")
    main()
