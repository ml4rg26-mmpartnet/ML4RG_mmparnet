"""Model-agnostic dataset for RBP binding prediction (Milestone 1: binary labels).

Returns raw biological data per (RNA-window, RBP) pair.
Architecture-specific formatting (tokenisation, one-hot, etc.) belongs in
the collate_fn or the model — not here.

Dataset layout
--------------
  windows  × n_rbps  samples total
  Each sample = one (600-nt RNA window, one RBP) pair.

Usage
-----
  from mmpartnet.data.rbp_binding import build_dataset
  ds = build_dataset(split="train")
  sample = ds[0]   # dict with keys documented in __getitem__
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import os
from mmpartnet import config

# Paths resolve through config's env-overridable shared roots (no user/home baked in): default to the
# /mnt/storage1 mount, override any single path with its env var, or the whole mount with ML4RG_STORAGE.
_MM = config.MMPARNET_DIR
_SH = config.SHARED_DIR
_DATA_PT = Path(os.environ.get(
    "ML4RG_BINDING_PT",
    _MM / "manually_gathered/600nt_windows.no-one-hot.stripped.binding/"
          "600nt_windows.no-one-hot.stripped.binding.narrowpeak_intersect/dataset.pt"))
_RBP_TSV = Path(os.environ.get(
    "ML4RG_FULL_RBP_TSV", _SH / "parnet-eclip/models-full-rbp-set/full_rbp_set.tsv"))
_FASTA = Path(os.environ.get(
    "ML4RG_HUMAN_FASTA", _MM / "manually_gathered/ProtT5_zenodo_datasets/human.fasta"))
_EMB_H5 = Path(os.environ.get(
    "ML4RG_PROTT5_H5", _MM / "manually_gathered/ProtT5_zenodo_datasets/reduced_embeddings_file.h5"))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RBPBindingDataset(Dataset):
    """
    One sample = one (RNA window, RBP) pair.

    __getitem__ returns
    -------------------
    sequence    : str         — 600-nt RNA window (raw; encode in collate_fn / model)
    rbp_name    : str         — e.g. "AARS_K562"
    rbp_gene    : str         — e.g. "AARS"
    protein_emb : Tensor(1024,) | None  — ProtT5 embedding, None if missing
    cell_type   : str         — e.g. "K562"
    window_meta : dict        — {"name": ..., "pad_side": ...} from dataset.pt
    rbp_idx     : int         — position in the 223-RBP binding vector
    label       : Tensor()    — scalar 0. or 1. (binary binding label)
    """

    def __init__(
        self,
        windows: list,
        rbp_order_df: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
        label_key: str = "binding",
    ):
        self.windows    = windows
        self.rbp_order  = rbp_order_df["rbp_ct"].tolist()   # ["AARS_K562", ...]
        self.rbp_genes  = rbp_order_df["rbp"].tolist()       # ["AARS", ...]
        self.cell_types = rbp_order_df["ct"].tolist()        # ["K562", ...]
        self.embeddings = embeddings
        self.label_key  = label_key
        self.n_rbps     = len(self.rbp_order)

    def __len__(self) -> int:
        return len(self.windows) * self.n_rbps

    def __getitem__(self, idx: int) -> dict:
        window_idx = idx // self.n_rbps
        rbp_idx    = idx  % self.n_rbps

        window    = self.windows[window_idx]
        rbp_name  = self.rbp_order[rbp_idx]
        gene      = self.rbp_genes[rbp_idx].upper()
        cell_type = self.cell_types[rbp_idx]

        sequence = window["inputs"]["sequence"]
        label    = window["outputs"][self.label_key][rbp_idx].float()

        emb_arr = self.embeddings.get(gene)
        protein_emb: Optional[torch.Tensor] = (
            torch.tensor(emb_arr, dtype=torch.float32) if emb_arr is not None else None
        )

        return {
            "sequence":    sequence,
            "rbp_name":    rbp_name,
            "rbp_gene":    gene,
            "protein_emb": protein_emb,
            "cell_type":   cell_type,
            "window_meta": window["meta"],
            "rbp_idx":     rbp_idx,
            "label":       label,
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_rbp_order(tsv_path: Path) -> pd.DataFrame:
    """Load RBP order TSV with columns [rbp_ct, rbp, ct]."""
    return pd.read_csv(tsv_path, sep="\t")


def parse_fasta_gene_index(fasta_path: Path) -> dict[str, int]:
    """Map UniProt gene name → sequence index in the FASTA file."""
    gene_to_idx: dict[str, int] = {}
    idx = 0
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                m = re.match(r">sp\|[A-Z0-9]+\|([A-Z0-9]+)_HUMAN", line)
                if m:
                    gene_to_idx[m.group(1)] = idx
                idx += 1
    return gene_to_idx


def load_rbp_embeddings(
    h5_path: Path,
    rbp_genes: list[str],
    gene_to_fasta_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    """Load ProtT5 embeddings for the requested RBP genes from an HDF5 file."""
    embeddings: dict[str, np.ndarray] = {}
    missing: list[str] = []

    unique_genes = {g.upper() for g in rbp_genes}
    with h5py.File(h5_path, "r") as f:
        for gene in unique_genes:
            if gene not in gene_to_fasta_idx:
                missing.append(gene)
                continue
            key = str(gene_to_fasta_idx[gene])
            if key not in f:
                missing.append(gene)
                continue
            embeddings[gene] = f[key][:]

    if missing:
        print(f"[rbp_binding] Warning: {len(missing)} RBPs without embedding: {missing[:5]}")
    return embeddings


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def build_dataset(split: str = "train", label_key: str = "binding") -> RBPBindingDataset:
    """Build an RBPBindingDataset for the given split ('train' | 'val' | 'test')."""
    print(f"[rbp_binding] Loading dataset (split='{split}') ...")

    data    = torch.load(_DATA_PT, mmap=True, weights_only=False)
    windows = data[split]

    rbp_df      = load_rbp_order(_RBP_TSV)
    gene_to_idx = parse_fasta_gene_index(_FASTA)
    embeddings  = load_rbp_embeddings(_EMB_H5, rbp_df["rbp"].tolist(), gene_to_idx)

    ds = RBPBindingDataset(windows, rbp_df, embeddings, label_key=label_key)
    print(
        f"[rbp_binding] Ready: {len(ds):,} samples "
        f"({len(windows)} windows × {len(rbp_df)} RBPs)"
    )
    return ds
