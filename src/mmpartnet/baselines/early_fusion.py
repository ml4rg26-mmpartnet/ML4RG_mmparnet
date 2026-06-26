"""Early-fusion baseline: concat[RNA_vec, ProtT5_vec] -> MLP -> P(bound).

One example = one (window, RBP) pair. The protein vector varies across the batch
(different RBPs in different examples), which forces the model to use the protein
channel to know WHICH RBP it is predicting for.

Inputs on disk (on the VM):
  ~/storage_ml4rg26-mmparnet/manually_gathered/
    600nt_windows.no-one-hot.stripped.binding/
      600nt_windows.no-one-hot.stripped.binding.narrowpeak_intersect/
        dataset.pt        # dict {train/test/valid: list of {inputs, outputs, meta}}
        rbp_cts.tsv       # RBP-column order for outputs.binding
    ProtT5_zenodo_datasets/
      reduced_embeddings_file.h5   # {idx_str: (1024,) float32}, idx = line number in fasta
      human.fasta                  # UniProt headers carry GN=<gene_symbol>

Companion notebook: notebooks/early_fusion_baseline.ipynb
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------- protein lookup

def build_gene_to_idx(fasta_path: str | Path) -> dict[str, str]:
    """Parse human.fasta -> {gene_symbol: h5_key_string}.

    The Zenodo ProtT5 h5 keys are stringified sequential indices into human.fasta
    ('0', '1', ...). UniProt headers carry the gene symbol in `GN=<symbol>`.
    First occurrence wins (UniProt has one canonical entry per gene most of the time).
    """
    gene_to_idx: dict[str, str] = {}
    idx = 0
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                m = re.search(r"GN=(\S+)", line)
                if m:
                    gene_to_idx.setdefault(m.group(1), str(idx))
                idx += 1
    return gene_to_idx


class ProteinEmbeddings:
    """Lazy ProtT5 lookup: gene symbol -> 1024-d vector. Caches reads in memory."""

    def __init__(self, h5_path: str | Path, gene_to_idx: dict[str, str]):
        self.h5_path = str(h5_path)
        self.gene_to_idx = gene_to_idx
        self._cache: dict[str, np.ndarray] = {}
        self._h5: Optional[h5py.File] = None

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def vector(self, gene: str) -> Optional[np.ndarray]:
        if gene in self._cache:
            return self._cache[gene]
        idx = self.gene_to_idx.get(gene)
        if idx is None:
            return None
        h5 = self._open()
        if idx not in h5:
            return None
        v = np.asarray(h5[idx], dtype=np.float32)
        self._cache[gene] = v
        return v


# ----------------------------------------------------------------- RNA encoding

_BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": -1}


def one_hot_meanpool(sequence: str) -> np.ndarray:
    """600-nt seq -> base-frequency vector (4-d). Trivial RNA encoder for the baseline.

    Swap for PARNET body features (512-d) once the pipeline works end-to-end.
    Mean-pooling discards positional info; if AUC is poor, that's the first thing to fix.
    """
    counts = np.zeros(4, dtype=np.float32)
    valid = 0
    for ch in sequence:
        i = _BASES.get(ch, -1)
        if i >= 0:
            counts[i] += 1
            valid += 1
    if valid > 0:
        counts /= valid
    return counts


# ---------------------------------------------------------------------- dataset

class EarlyFusionDataset(Dataset):
    """One example = (rna_vec, protein_vec, label) for a single (window, RBP) pair.

    We expand each window into N_rbps pairs lazily via flat indexing in __getitem__.

    Args:
      records:    one split of dataset.pt, e.g. d["train"]
      rbps:       column-aligned RBP symbols from rbp_cts.tsv (binding[i] -> rbps[i])
      proteins:   ProteinEmbeddings for symbol -> ProtT5 lookup
      rna_encoder: seq str -> np.ndarray
      rbp_subset: restrict to these RBP symbols (typically the ProtT5-covered subset)
    """

    def __init__(
        self,
        records: list,
        rbps: list[str],
        proteins: ProteinEmbeddings,
        rna_encoder=one_hot_meanpool,
        rbp_subset: Optional[list[str]] = None,
    ):
        self.records = records
        if rbp_subset is not None:
            keep = set(rbp_subset)
            self.rbp_cols = [i for i, g in enumerate(rbps) if g in keep]
            self.rbps = [rbps[i] for i in self.rbp_cols]
        else:
            self.rbp_cols = list(range(len(rbps)))
            self.rbps = list(rbps)
        self.proteins = proteins
        self.rna_encoder = rna_encoder
        self.N_rbps = len(self.rbps)

    def __len__(self):
        return len(self.records) * self.N_rbps

    def __getitem__(self, idx):
        rec_i, rbp_j = divmod(idx, self.N_rbps)
        rec = self.records[rec_i]
        seq = rec["inputs"]["sequence"]
        binding = rec["outputs"]["binding"]
        rbp = self.rbps[rbp_j]
        col = self.rbp_cols[rbp_j]

        rna_vec = self.rna_encoder(seq)
        prot_vec = self.proteins.vector(rbp)
        if prot_vec is None:
            prot_vec = np.zeros(1024, dtype=np.float32)  # shouldn't happen if rbp_subset filtered
        label = float(binding[col].item() > 0)
        return (
            torch.from_numpy(rna_vec),
            torch.from_numpy(prot_vec),
            torch.tensor(label, dtype=torch.float32),
        )


# ------------------------------------------------------------------------ model

class EarlyFusion(nn.Module):
    """concat(RNA, protein) -> 2-hidden-layer MLP -> 1 logit."""

    def __init__(self, rna_dim: int = 4, prot_dim: int = 1024, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rna_dim + prot_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, rna: torch.Tensor, prot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([rna, prot], dim=-1)).squeeze(-1)


# --------------------------------------------------------------- training / eval

def train_one_epoch(model, loader, optim, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for rna, prot, y in loader:
        rna, prot, y = rna.to(device), prot.to(device), y.to(device)
        logits = model(rna, prot)
        loss = loss_fn(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * y.numel()
        n += y.numel()
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.eval()
    ys, ps = [], []
    for rna, prot, y in loader:
        rna, prot = rna.to(device), prot.to(device)
        p = torch.sigmoid(model(rna, prot)).cpu().numpy()
        ys.append(y.numpy())
        ps.append(p)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    has_both = len(np.unique(y)) > 1
    return {
        "auroc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if has_both else float("nan"),
        "pos_rate": float(y.mean()),
        "n": int(len(y)),
    }


def protein_shuffle_mapping(gene_to_idx: dict[str, str], seed: int = 0) -> dict[str, str]:
    """Permute the {gene -> h5 idx} mapping. The mandatory control: if AUC stays
    the same under this, the model isn't using the protein channel and the result is fake.
    """
    rng = np.random.default_rng(seed)
    genes = list(gene_to_idx.keys())
    idxs = list(gene_to_idx.values())
    rng.shuffle(idxs)
    return dict(zip(genes, idxs))
