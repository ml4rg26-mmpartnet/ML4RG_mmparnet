#!/usr/bin/env python
"""Map PARNET RBP-cell tracks to pooled ProtT5 embeddings.

The PARNET track table uses RBP gene symbols such as ``QKI`` and track names such
as ``QKI_HepG2``.  The ProtT5 Zenodo files store embeddings by numeric HDF5 key,
matching the order of records in ``human.fasta``.  This script bridges those two
worlds and writes a track-level mapping table.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import h5py


SHARED = Path("/home/dgu/storage_ml4rg26-shared")
MMPARNET = Path("/home/dgu/storage_ml4rg26-mmparnet")
DEFAULT_TRACKS = SHARED / "parnet-eclip/models-full-rbp-set/full_rbp_set.tsv"
DEFAULT_PROTT5 = MMPARNET / "manually_gathered/ProtT5_zenodo_datasets"
DEFAULT_FASTA = DEFAULT_PROTT5 / "human.fasta"
DEFAULT_H5 = DEFAULT_PROTT5 / "reduced_embeddings_file.h5"

# Legacy ENCODE/PARNET symbols that differ from the UniProt GN= symbol in human.fasta.
GENE_ALIASES = {
    "AARS": "AARS1",
    "TROVE2": "RO60",
}


def parse_fasta_headers(path: Path) -> tuple[list[dict[str, str]], dict[str, list[int]]]:
    records: list[dict[str, str]] = []
    gene_to_indices: dict[str, list[int]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            parts = header.split(maxsplit=1)
            original_id = parts[0]
            description = parts[1] if len(parts) > 1 else ""
            fields = original_id.split("|")
            uniprot_acc = fields[1] if len(fields) > 1 else original_id
            uniprot_entry = fields[2] if len(fields) > 2 else ""
            gene_match = re.search(r"\bGN=([^\s]+)", header)
            gene = gene_match.group(1) if gene_match else ""

            record = {
                "fasta_index": str(len(records)),
                "h5_key": str(len(records)),
                "original_id": original_id,
                "uniprot_acc": uniprot_acc,
                "uniprot_entry": uniprot_entry,
                "gene": gene,
                "description": description,
            }
            records.append(record)
            if gene:
                gene_to_indices.setdefault(gene, []).append(int(record["fasta_index"]))
    return records, gene_to_indices


def load_tracks(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            rows.append({
                "track_index": str(i),
                "rbp_ct": row["rbp_ct"],
                "rbp": row["rbp"],
                "ct": row["ct"],
            })
    return rows


def choose_match(rbp: str, gene_to_indices: dict[str, list[int]]) -> tuple[str | None, str, str]:
    if rbp in gene_to_indices:
        return rbp, "direct", ""
    alias = GENE_ALIASES.get(rbp)
    if alias and alias in gene_to_indices:
        return alias, "alias", f"{rbp}->{alias}"
    return None, "missing", ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/dgu/workspace/ML4RG_mmparnet/mmpartnet_out/prott5_track_map.tsv"),
    )
    args = parser.parse_args()

    tracks = load_tracks(args.tracks)
    records, gene_to_indices = parse_fasta_headers(args.fasta)

    with h5py.File(args.h5, "r") as h5:
        rows = []
        for track in tracks:
            match_gene, match_type, note = choose_match(track["rbp"], gene_to_indices)
            if match_gene is None:
                rows.append({
                    **track,
                    "status": "missing",
                    "match_type": match_type,
                    "match_gene": "",
                    "fasta_index": "",
                    "h5_key": "",
                    "embedding_dim": "",
                    "original_id": "",
                    "uniprot_acc": "",
                    "uniprot_entry": "",
                    "fasta_gene": "",
                    "description": "",
                    "note": note,
                })
                continue

            indices = gene_to_indices[match_gene]
            fasta_index = indices[0]
            rec = records[fasta_index]
            h5_key = rec["h5_key"]
            h5_exists = h5_key in h5
            rows.append({
                **track,
                "status": "matched" if h5_exists else "missing_h5_key",
                "match_type": match_type,
                "match_gene": match_gene,
                "fasta_index": rec["fasta_index"],
                "h5_key": h5_key,
                "embedding_dim": str(h5[h5_key].shape[0]) if h5_exists else "",
                "original_id": rec["original_id"],
                "uniprot_acc": rec["uniprot_acc"],
                "uniprot_entry": rec["uniprot_entry"],
                "fasta_gene": rec["gene"],
                "description": rec["description"],
                "note": note,
            })

    fieldnames = [
        "track_index",
        "rbp_ct",
        "rbp",
        "ct",
        "status",
        "match_type",
        "match_gene",
        "fasta_index",
        "h5_key",
        "embedding_dim",
        "original_id",
        "uniprot_acc",
        "uniprot_entry",
        "fasta_gene",
        "description",
        "note",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(row["status"] == "matched" for row in rows)
    alias = sum(row["match_type"] == "alias" for row in rows)
    unique_rbps = sorted({row["rbp"] for row in tracks})
    unique_matched = {
        row["rbp"] for row in rows
        if row["status"] == "matched"
    }
    print(f"tracks:              {len(rows)}")
    print(f"unique rbps:         {len(unique_rbps)}")
    print(f"matched tracks:      {matched}/{len(rows)}")
    print(f"matched unique rbps: {len(unique_matched)}/{len(unique_rbps)}")
    print(f"alias-matched rows:  {alias}")
    print(f"wrote:               {args.out}")


if __name__ == "__main__":
    main()
