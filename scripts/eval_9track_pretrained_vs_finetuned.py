#!/usr/bin/env python
"""Evaluate the 9-track spliceosome-HepG2 baseline on length-600 test windows.

This deliberately starts with the simplest reproducible slice:
- use only samples whose sequence length is already 600, so no padding/mask logic is needed;
- compare the full 223-track pretrained PARNET, restricted to the 9 demo tracks, against the
  teacher-provided 9-track finetuned head.
"""
from __future__ import annotations

import argparse
import csv
import copy
import os
from pathlib import Path

import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REFS = REPO.parent / "parnet_refs"
SHARED = Path("/home/dgu/storage_ml4rg26-shared")
DEFAULT_WEIGHTS = SHARED / "parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt"
DEFAULT_DATASET = SHARED / "parnet-demo/spliceosome-hepg2.precomputed/datasets/dataset.pt"
DEFAULT_TRACKS = SHARED / "parnet-demo/spliceosome-hepg2.precomputed/datasets/rbp_cts.tsv"
DEFAULT_FINETUNED = (
    SHARED
    / "parnet-demo/spliceosome-hepg2.precomputed/training/"
    / "parnet.7m-0.0.ft-head.spliceosome-hepg2/model.statedict.pt"
)


class LinearProjectionHead(nn.Module):
    def __init__(self, n_tracks: int):
        super().__init__()
        self.pointwise = nn.Conv1d(512, n_tracks, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(x)


class MixCoeffMLP(nn.Module):
    def __init__(self, n_tracks: int, units: int = 128):
        super().__init__()
        self.dense1 = nn.Linear(512, units)
        self.act = nn.ReLU()
        self.dense2 = nn.Linear(units, n_tracks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=-1)
        return torch.sigmoid(self.dense2(self.act(self.dense1(x))))


class NineTrackHead(nn.Module):
    """Matches the teacher finetuned checkpoint key names under ``head.*``."""

    def __init__(self, n_tracks: int = 9):
        super().__init__()
        self.head_target = LinearProjectionHead(n_tracks)
        self.head_control = LinearProjectionHead(n_tracks)
        self.mix_coeff = MixCoeffMLP(n_tracks)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        target_logits = self.head_target(x)
        control_logits = self.head_control(x)
        mix = self.mix_coeff(x).unsqueeze(-1)

        target_logprob = target_logits - torch.logsumexp(target_logits, dim=-1, keepdim=True)
        control_logprob = control_logits - torch.logsumexp(control_logits, dim=-1, keepdim=True)
        max_logprob = torch.maximum(target_logprob, control_logprob)
        total_logprob = max_logprob + torch.log(
            mix * torch.exp(target_logprob - max_logprob)
            + (1.0 - mix) * torch.exp(control_logprob - max_logprob)
        )
        return {
            "target": target_logprob,
            "control": control_logprob,
            "total": total_logprob,
            "mix_coeff": mix.squeeze(-1),
        }


class FineTunedNineTrackParnet(nn.Module):
    def __init__(self, pretrained_module: nn.Module):
        super().__init__()
        self.stem = copy.deepcopy(pretrained_module.stem)
        self.body = copy.deepcopy(pretrained_module.body)
        self.projection = copy.deepcopy(getattr(pretrained_module, "projection", None))
        self.head = NineTrackHead(9)

    def forward(self, onehot: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(onehot)
        x = self.body(x)
        if self.projection is not None:
            x = self.projection(x)
        return self.head(x)


def onehot(seq: str) -> torch.Tensor:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    x = torch.zeros(4, len(seq), dtype=torch.float32)
    for j, ch in enumerate(seq.upper()):
        idx = base_to_idx.get(ch)
        if idx is not None:
            x[idx, j] = 1.0
    return x


def sparse_to_dense(sp: dict) -> torch.Tensor:
    return torch.sparse_coo_tensor(sp["indices"], sp["values"], sp["size"]).to_dense().float()


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if float(denom) <= 1e-9:
        return float("nan")
    return float((x * y).sum() / denom)


def load_track_table(path: Path) -> tuple[list[str], list[int]]:
    names: list[str] = []
    full_indices: list[int] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            names.append(row["rbp_ct"])
            full_indices.append(int(row["track_index_in_full_dataset"]))
    return names, full_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-count", type=float, default=10.0)
    args = parser.parse_args()

    os.environ.setdefault("ML4RG_REFS", str(DEFAULT_REFS))
    os.environ.setdefault("ML4RG_PARNET_WEIGHTS", str(DEFAULT_WEIGHTS))

    from mmpartnet.models.parnet import load_parnet

    track_names, full_indices = load_track_table(DEFAULT_TRACKS)
    pretrained = load_parnet(device="cpu")

    finetuned = FineTunedNineTrackParnet(pretrained.m)
    ft_payload = torch.load(DEFAULT_FINETUNED, map_location="cpu", weights_only=False)
    finetuned.load_state_dict(ft_payload["state_dict"], strict=True)
    finetuned.eval()

    data = torch.load(DEFAULT_DATASET, map_location="cpu", weights_only=False)[args.split]
    selected = [x for x in data if len(x["inputs"]["sequence"]) == 600][: args.max_windows]
    print(f"split={args.split} selected_length_600={len(selected)} max_windows={args.max_windows}")
    print("tracks:", ", ".join(track_names))
    print("full pretrained track indices:", full_indices)

    per_track_pre: list[list[float]] = [[] for _ in track_names]
    per_track_ft: list[list[float]] = [[] for _ in track_names]

    with torch.no_grad():
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start : start + args.batch_size]
            seqs = [x["inputs"]["sequence"] for x in batch]
            xb = torch.stack([onehot(s) for s in seqs])

            pre_total = pretrained.full(xb)["total"][:, full_indices, :]
            ft_total = torch.softmax(finetuned(xb)["total"], dim=-1)
            obs = torch.stack([sparse_to_dense(x["outputs"]["eCLIP"]) for x in batch])

            for b in range(obs.shape[0]):
                for t in range(len(track_names)):
                    counts = obs[b, t]
                    if float(counts.sum()) < args.min_count:
                        continue
                    true_profile = counts / counts.sum()
                    per_track_pre[t].append(pearson(pre_total[b, t], true_profile))
                    per_track_ft[t].append(pearson(ft_total[b, t], true_profile))

    print("\nRBP_track\tN\tpretrained_Pearson\tfinetuned_Pearson\tdelta")
    pre_means = []
    ft_means = []
    for name, pre_vals, ft_vals in zip(track_names, per_track_pre, per_track_ft):
        pre = torch.tensor(pre_vals).nanmean().item() if pre_vals else float("nan")
        ft = torch.tensor(ft_vals).nanmean().item() if ft_vals else float("nan")
        pre_means.append(pre)
        ft_means.append(ft)
        print(f"{name}\t{len(pre_vals)}\t{pre:+.4f}\t{ft:+.4f}\t{ft - pre:+.4f}")

    mean_pre = torch.tensor(pre_means).nanmean().item()
    mean_ft = torch.tensor(ft_means).nanmean().item()
    print(f"\nMEAN\t{sum(len(v) for v in per_track_pre)}\t{mean_pre:+.4f}\t{mean_ft:+.4f}\t{mean_ft - mean_pre:+.4f}")


if __name__ == "__main__":
    main()
