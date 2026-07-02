"""mmseqs2 sequence-identity family clustering — the CORRECT family metric for the interpolation-in-
RBP-space analysis (domain-type over-merges; pooled-ESM cosine is anisotropy-corrupted). On ALL project
proteins (CORAL 3138 + Moyon eCLIP/affinity 263) this yields ~1744 total families / 99 with specificity
labels, which overturned the pessimistic "n=4 wells" and makes the family-scaling lever data-available.

Two entry points:
  load_clusters(tsv)  -> {member: representative}   ALWAYS available; loads a precomputed mmseqs
                         `*_cluster.tsv` (how notebooks consume the pinned family map).
  cluster_fasta(fa)   -> {member: representative}   runs mmseqs if the binary is available (arg / env
                         ML4RG_MMSEQS / PATH), else raises with the exact CLI to run on a node.
`min_seq_id`/`coverage` are PINNED here so the family count is reproducible (drift moves the curve)."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

MIN_SEQ_ID = 0.30   # pinned: 30% identity = the family granularity behind the 1744/99 count
COVERAGE = 0.80


def load_clusters(tsv) -> dict:
    """Load an mmseqs `*_cluster.tsv` (representative<TAB>member per line) -> {member: representative}."""
    out = {}
    with open(tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[1]] = parts[0]
    return out


def n_families(cluster_map) -> int:
    """Distinct families (representatives) in a {member: representative} map."""
    return len(set(cluster_map.values()))


def cluster_fasta(fasta, min_seq_id=MIN_SEQ_ID, coverage=COVERAGE, tmp=None, mmseqs=None) -> dict:
    """Run `mmseqs easy-cluster` -> {member: representative}. Needs the mmseqs binary (arg `mmseqs`, or
    $ML4RG_MMSEQS, or 'mmseqs' on PATH). Raises RuntimeError with the CLI if unavailable."""
    exe = mmseqs or os.environ.get("ML4RG_MMSEQS", "mmseqs")
    with tempfile.TemporaryDirectory() as _td:
        td = tmp or _td
        pref = str(Path(td) / "clust")
        cmd = [exe, "easy-cluster", str(fasta), pref, str(Path(td) / "mmtmp"),
               "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"mmseqs unavailable/failed ({type(e).__name__}); run on a node:\n  "
                               + " ".join(cmd)) from e
        return load_clusters(pref + "_cluster.tsv")
