#!/usr/bin/env python
"""Score exported cross-attention interpretation records against motif matrices.

The interpretation export already contains per-position distributions. This
script is intentionally post hoc: it parses TRANSFAC-style motif matrices from
tarballs/directories, builds sequence masks for each exported sample, and writes
motif overlap metrics without re-running the model.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


@dataclass(frozen=True)
class MotifMatrix:
    source: str
    rbp: str
    motif_id: str
    motif_type: str
    path: str
    matrix: torch.Tensor


def normalize_rbp(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def parse_transfac_text(text: str, path: str) -> MotifMatrix | None:
    motif_id = Path(path).name
    motif_type = ""
    rbp = ""
    source = Path(path).parts[0] if Path(path).parts else ""
    alphabet: list[str] | None = None
    rows: list[list[float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in {"XX", "//"}:
            continue
        parts = line.split()
        key = parts[0]
        if key == "ID" and len(parts) > 1:
            motif_id = parts[1]
        elif key == "MT" and len(parts) > 1:
            motif_type = parts[1].upper()
        elif key == "NA" and len(parts) > 1:
            rbp = parts[1]
        elif key == "DB" and len(parts) > 1:
            source = parts[1]
        elif key == "P0" and len(parts) >= 5:
            alphabet = [base.upper() for base in parts[1:]]
        elif key.isdigit() and alphabet is not None and len(parts) >= len(alphabet) + 1:
            values = [float(value) for value in parts[1 : len(alphabet) + 1]]
            acgu = [0.0, 0.0, 0.0, 0.0]
            for base, value in zip(alphabet, values):
                if base in BASE_TO_INDEX:
                    acgu[BASE_TO_INDEX[base]] += value
            rows.append(acgu)

    if not rbp or not rows:
        return None
    matrix = torch.tensor(rows, dtype=torch.float32)
    if motif_type in {"PPM", "PCM"}:
        matrix = matrix.clamp_min(0.0)
        row_sums = matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        matrix = matrix / row_sums
    return MotifMatrix(source=source, rbp=rbp, motif_id=motif_id, motif_type=motif_type, path=path, matrix=matrix)


def iter_transfac_files(path: Path) -> list[MotifMatrix]:
    motifs: list[MotifMatrix] = []
    if path.is_file() and (path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz"):
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".transfac"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                text = io.TextIOWrapper(extracted, encoding="utf-8", errors="replace").read()
                motif = parse_transfac_text(text, member.name)
                if motif is not None:
                    motifs.append(motif)
    elif path.is_dir():
        for motif_path in path.rglob("*.transfac"):
            text = motif_path.read_text(encoding="utf-8", errors="replace")
            motif = parse_transfac_text(text, str(motif_path))
            if motif is not None:
                motifs.append(motif)
    else:
        raise ValueError(f"unsupported motif input: {path}")
    return motifs


def load_motif_index(paths: list[Path], motif_types: set[str]) -> dict[str, list[MotifMatrix]]:
    index: dict[str, list[MotifMatrix]] = {}
    for path in paths:
        for motif in iter_transfac_files(path):
            if motif_types and motif.motif_type not in motif_types:
                continue
            index.setdefault(normalize_rbp(motif.rbp), []).append(motif)
    return index


def motif_hits_mask(seq: str, motif: MotifMatrix, score_fraction: float) -> tuple[torch.Tensor, int]:
    rna = seq.upper().replace("T", "U")
    length = len(rna)
    width = int(motif.matrix.shape[0])
    mask = torch.zeros(length, dtype=torch.bool)
    if width <= 0 or length < width:
        return mask, 0

    matrix = motif.matrix
    max_score = float(matrix.max(dim=1).values.sum())
    threshold = score_fraction * max_score
    hit_count = 0
    for start in range(0, length - width + 1):
        score = 0.0
        ok = True
        for offset, base in enumerate(rna[start : start + width]):
            idx = BASE_TO_INDEX.get(base)
            if idx is None:
                ok = False
                break
            score += float(matrix[offset, idx])
        if ok and score >= threshold:
            mask[start : start + width] = True
            hit_count += 1
    return mask, hit_count


def merged_motif_mask(seq: str, motifs: list[MotifMatrix], score_fraction: float) -> tuple[torch.Tensor, int]:
    mask = torch.zeros(len(seq), dtype=torch.bool)
    hit_count = 0
    for motif in motifs:
        motif_mask, motif_hits = motif_hits_mask(seq, motif, score_fraction)
        mask[: motif_mask.numel()] |= motif_mask
        hit_count += motif_hits
    return mask, hit_count


def prob_on_mask(prob: torch.Tensor | None, mask: torch.Tensor) -> float | None:
    if prob is None:
        return None
    usable_mask = mask[: prob.numel()].to(dtype=prob.dtype)
    return float((prob * usable_mask).sum())


def topk_overlap(prob: torch.Tensor | None, mask: torch.Tensor, k: int) -> float | None:
    if prob is None:
        return None
    k = min(k, prob.numel())
    usable_mask = mask[: prob.numel()]
    if k <= 0:
        return None
    top_idx = prob.topk(k).indices
    return float(usable_mask[top_idx].to(dtype=torch.float32).mean())


def tensor_or_none(record: dict, key: str) -> torch.Tensor | None:
    value = record.get(key)
    if value is None:
        return None
    return value.detach().cpu().float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpretation", type=Path, required=True, help="Output .pt from export_cross_attention_interpretation.py")
    parser.add_argument("--motif-input", type=Path, action="append", required=True, help="Motif tar.gz or directory; repeatable.")
    parser.add_argument("--motif-type", action="append", default=None, help="TRANSFAC MT type to use; repeatable. Default: PPM.")
    parser.add_argument("--score-fraction", type=float, default=0.8, help="Hit threshold as a fraction of each motif's best possible PPM score.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.interpretation, map_location="cpu", weights_only=False)
    motif_types = {"PPM"} if args.motif_type is None else {motif_type.upper() for motif_type in args.motif_type}
    motif_index = load_motif_index(args.motif_input, motif_types)
    rows = []
    for record in payload["records"]:
        rbp = str(record["rbp"])
        motifs = motif_index.get(normalize_rbp(rbp), [])
        motif_mask, hit_count = merged_motif_mask(str(record["sequence"]), motifs, args.score_fraction)
        target = tensor_or_none(record, "target_prob")
        binary = tensor_or_none(record, "binary_position_prob")
        alpha = tensor_or_none(record, "alpha_bind")
        row = {
            "rbp": rbp,
            "cell": record["cell"],
            "window_index": record["window_index"],
            "track_index": record["track_index"],
            "binding_label": record["binding_label"],
            "binding_prob": record.get("binding_prob"),
            "binding_gate": record.get("binding_gate"),
            "motif_count": len(motifs),
            "motif_hit_count": hit_count,
            "has_motif": int(bool(motif_mask.any())),
            "motif_covered_bases": int(motif_mask.sum()),
            "target_prob_on_motif": prob_on_mask(target, motif_mask),
            "binary_prob_on_motif": prob_on_mask(binary, motif_mask),
            "alpha_bind_on_motif": prob_on_mask(alpha, motif_mask),
            "topk_target_overlap_motif": topk_overlap(target, motif_mask, args.topk),
            "topk_binary_overlap_motif": topk_overlap(binary, motif_mask, args.topk),
            "topk_alpha_overlap_motif": topk_overlap(alpha, motif_mask, args.topk),
        }
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "rbp",
            "cell",
            "window_index",
            "track_index",
            "binding_label",
            "binding_prob",
            "binding_gate",
            "motif_count",
            "motif_hit_count",
            "has_motif",
            "motif_covered_bases",
            "target_prob_on_motif",
            "binary_prob_on_motif",
            "alpha_bind_on_motif",
            "topk_target_overlap_motif",
            "topk_binary_overlap_motif",
            "topk_alpha_overlap_motif",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    motif_total = sum(len(value) for value in motif_index.values())
    print(f"loaded motifs: {motif_total} across {len(motif_index)} RBPs")
    print(f"wrote motif metrics: {args.out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mmpartnet_pycache")
    main()
