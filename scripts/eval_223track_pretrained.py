#!/usr/bin/env python
"""Evaluate the full 223-track pretrained PARNET on held-out windows.

This is the full-dataset counterpart of the 9-track sanity script.  It deliberately
starts with the safest path: only windows whose sequence is already exactly 600 nt
are evaluated, so we do not need padding/mask logic yet.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_from_disk


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REFS = REPO.parent / "parnet_refs"
SHARED = Path("/home/dgu/storage_ml4rg26-shared")
DEFAULT_WEIGHTS = SHARED / "parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt"
DEFAULT_HFDS = (
    SHARED
    / "parnet-eclip/data-formatted-for-training/"
    / "600nt_windows.no-one-hot.stripped/encode.filtered.hfds"
)


def sparse_to_dense(sp: dict) -> torch.Tensor:
    if not sp["values"]:
        return torch.zeros(tuple(sp["size"]), dtype=torch.float32)
    return torch.sparse_coo_tensor(sp["indices"], sp["values"], sp["size"]).to_dense().float()


def iter_exact_length_batches(dataset, batch_size: int, max_windows: int | None, seq_len: int):
    batch = []
    seen = 0
    skipped = 0
    for sample in dataset:
        seq = sample["inputs"]["sequence"]
        if len(seq) != seq_len:
            skipped += 1
            continue
        batch.append(sample)
        seen += 1
        if len(batch) == batch_size:
            yield batch, seen, skipped
            batch = []
        if max_windows is not None and seen >= max_windows:
            break
    if batch:
        yield batch, seen, skipped


def update_pearson_stats(
    pred: torch.Tensor,
    counts: torch.Tensor,
    min_count: float,
    corr_sum: torch.Tensor,
    corr_n: torch.Tensor,
) -> None:
    """Accumulate per-track Pearson(pred_profile, true_count_profile)."""
    count_sum = counts.sum(dim=-1)
    true = counts / count_sum.clamp_min(1.0).unsqueeze(-1)

    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    true_centered = true - true.mean(dim=-1, keepdim=True)
    numerator = (pred_centered * true_centered).sum(dim=-1)
    denominator = torch.sqrt(
        (pred_centered * pred_centered).sum(dim=-1)
        * (true_centered * true_centered).sum(dim=-1)
    )
    valid = (count_sum >= min_count) & (denominator > 1e-9)
    corr = numerator / denominator.clamp_min(1e-9)

    corr_sum += torch.where(valid, corr, torch.zeros_like(corr)).sum(dim=0).cpu()
    corr_n += valid.sum(dim=0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-windows", type=int, default=256, help="Number of exact-length windows to evaluate; 0 means all.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=600)
    parser.add_argument("--min-count", type=float, default=10.0)
    parser.add_argument("--prediction", default="total", choices=["total", "target", "control"])
    parser.add_argument("--label", default="eCLIP", choices=["eCLIP", "control"])
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--hfds", type=Path, default=DEFAULT_HFDS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Optional path prefix for JSON/TSV outputs. If omitted, only print metrics.",
    )
    parser.add_argument("--progress-every", type=int, default=256)
    args = parser.parse_args()
    max_windows = None if args.max_windows == 0 else args.max_windows

    os.environ.setdefault("ML4RG_REFS", str(DEFAULT_REFS))
    os.environ["ML4RG_PARNET_WEIGHTS"] = str(args.weights)

    from mmpartnet.models.parnet import load_parnet
    from mmpartnet.process.onehot import batch_onehot

    model = load_parnet(weights=args.weights, device=args.device)
    data = load_from_disk(str(args.hfds))[args.split]

    n_tracks = len(model.syms)
    corr_sum = torch.zeros(n_tracks, dtype=torch.float64)
    corr_n = torch.zeros(n_tracks, dtype=torch.long)
    processed = 0
    skipped = 0
    started = time.time()

    print(f"weights:    {args.weights}")
    print(f"dataset:    {args.hfds}")
    print(f"split:      {args.split} ({len(data)} rows)")
    print(f"prediction: {args.prediction}")
    print(f"label:      {args.label}")
    print(f"device:     {model.device}")
    print(f"tracks:     {n_tracks}")
    print(f"max_windows={max_windows or 'all'} batch_size={args.batch_size} min_count={args.min_count}")

    with torch.no_grad():
        for batch, seen, skipped_now in iter_exact_length_batches(
            data, args.batch_size, max_windows, args.seq_len
        ):
            skipped = skipped_now
            seqs = [sample["inputs"]["sequence"] for sample in batch]
            xb = batch_onehot(seqs, device=model.device)
            pred = model.full(xb)[args.prediction].detach().cpu()
            counts = torch.stack([sparse_to_dense(sample["outputs"][args.label]) for sample in batch])
            update_pearson_stats(pred, counts, args.min_count, corr_sum, corr_n)

            processed = seen
            if args.progress_every and processed % args.progress_every == 0:
                elapsed = time.time() - started
                print(f"processed={processed} skipped_short={skipped} elapsed_sec={elapsed:.1f}")

    per_track = []
    means = corr_sum / corr_n.clamp_min(1)
    for i, ((symbol, cell), n) in enumerate(zip(model.syms, corr_n.tolist())):
        per_track.append(
            {
                "track_index": i,
                "symbol": symbol,
                "cell": cell,
                "n_profiles": int(n),
                "pearson": float(means[i]) if n else None,
            }
        )

    valid_means = means[corr_n > 0]
    macro_mean = float(valid_means.mean()) if len(valid_means) else float("nan")
    micro_mean = float(corr_sum.sum() / corr_n.sum().clamp_min(1))
    summary = {
        "weights": str(args.weights),
        "dataset": str(args.hfds),
        "split": args.split,
        "prediction": args.prediction,
        "label": args.label,
        "seq_len": args.seq_len,
        "min_count": args.min_count,
        "processed_windows": processed,
        "skipped_short_windows": skipped,
        "tracks_with_profiles": int((corr_n > 0).sum()),
        "profile_evaluations": int(corr_n.sum()),
        "macro_mean_pearson": macro_mean,
        "micro_mean_pearson": micro_mean,
        "elapsed_sec": time.time() - started,
        "per_track": per_track,
    }

    json_path = None
    tsv_path = None
    if args.out_prefix is not None:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = args.out_prefix.with_suffix(".json")
        tsv_path = args.out_prefix.with_suffix(".tsv")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with tsv_path.open("w", encoding="utf-8") as f:
            f.write("track_index\tsymbol\tcell\tn_profiles\tpearson\n")
            for row in per_track:
                pearson = "" if row["pearson"] is None else f"{row['pearson']:.6f}"
                f.write(
                    f"{row['track_index']}\t{row['symbol']}\t{row['cell']}\t"
                    f"{row['n_profiles']}\t{pearson}\n"
                )

    print("\nRESULT")
    print(f"processed windows:       {processed}")
    print(f"skipped short windows:   {skipped}")
    print(f"tracks with profiles:    {summary['tracks_with_profiles']}/{n_tracks}")
    print(f"profile evaluations:     {summary['profile_evaluations']}")
    print(f"macro mean Pearson:      {macro_mean:+.4f}")
    print(f"micro mean Pearson:      {micro_mean:+.4f}")
    if json_path is not None and tsv_path is not None:
        print(f"wrote JSON:              {json_path}")
        print(f"wrote TSV:               {tsv_path}")


if __name__ == "__main__":
    main()
