"""hg38 access + window-sequence extraction (wraps pyfaidx). Consolidates the window_seq
logic from eclip_m1_gpu/rna_structure_gate."""
from __future__ import annotations
from functools import lru_cache
from .. import config

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@lru_cache(maxsize=1)
def genome():
    from pyfaidx import Fasta
    return Fasta(str(config.HG38), sequence_always_upper=True)


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def window_seq(chrom, start, end, strand="+") -> str | None:
    """Sequence for [start,end) on the given strand (revcomp on '-'). None if out of bounds."""
    fa = genome()
    if chrom not in fa:
        return None
    s = max(0, int(start)); e = min(len(fa[chrom]), int(end))
    if e - s < 1:
        return None
    seq = str(fa[chrom][s:e])
    return revcomp(seq) if strand == "-" else seq


def centered_window(chrom, pstart, pend, strand, lwin) -> str | None:
    """Fixed-length window of length `lwin` centered on a peak interval."""
    c = (int(pstart) + int(pend)) // 2
    s = c - lwin // 2
    seq = window_seq(chrom, s, s + lwin, strand)
    if seq is None or len(seq) != lwin:
        return None
    return seq
