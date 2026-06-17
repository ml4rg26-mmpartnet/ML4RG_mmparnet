"""Regenerate the protein-rep npz files into data/ (reproducible, never committed).

Produces (consumed by io/embeddings.ProteinRep):
  data/embeddings_all.npz   ESM2-650M pooled (640-d) per RBP symbol
  data/pe_string.npz        STRING-PPI personalized-PageRank positional encoding
  data/perres64.npz         per-residue ESM (optional, for RIBEX-proxy attention)

This is the `protein='esm650_pooled'/'ribex_proxy'` substrate. The real RIBEX rep
(`protein='ribex_real'`) is a drop-in at io/embeddings.ProteinRep — no code change here.

Heavy deps (fair-esm / transformers) are OPTIONAL extras: `pixi run -e esm ...` or
`pip install -e '.[esm]'`. STRING-PE needs the STRING human PPI edge list.

Usage:
  python scripts/build_embeddings.py --rbps data/rbp_list.txt
"""
from __future__ import annotations
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rbps", help="text file of RBP gene symbols (one per line)")
    ap.add_argument("--out-dir", default=None, help="default: config.DATA")
    args = ap.parse_args()

    print(
        "build_embeddings.py is a documented stub.\n"
        "  1. load RBP symbols -> UniProt sequences\n"
        "  2. ESM2-650M (fair-esm) -> mean-pool -> embeddings_all.npz  (640-d per RBP)\n"
        "  3. STRING human PPI -> personalized PageRank per RBP -> pe_string.npz\n"
        "  4. (optional) per-residue ESM -> perres64.npz\n"
        "Install the heavy extras first:  pip install -e '.[esm]'  (or pixi -e esm).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
