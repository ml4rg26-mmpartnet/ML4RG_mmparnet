"""Local eCLIP peaks -> ParnetDataElement (the dev substrate, available now).
Reads the 234 local ENCODE peak BED.gz files (via eclip_manifest.json), centers a fixed
window on each peak, extracts the hg38 sequence, and emits the lab's ParnetDataElement
contract. Also provides N1/N2 negative sampling for M1.
"""
from __future__ import annotations
import gzip
import json
import numpy as np
from pathlib import Path

from .. import config, contract
from ..io import genome


def _load_manifest():
    return json.loads((config.DATA / "eclip_manifest.json").read_text())


def read_bed(path, max_peaks=100_000):
    """yield (chrom, start, end, strand, score) from an ENCODE eCLIP narrowPeak bed.gz."""
    out = []
    try:
        with gzip.open(path, "rt") as fh:
            for i, ln in enumerate(fh):
                if i >= max_peaks:
                    break
                f = ln.rstrip("\n").split("\t")
                if len(f) < 3:
                    continue
                strand = f[5] if len(f) > 5 and f[5] in "+-" else "+"
                score = float(f[6]) if len(f) > 6 and f[6] not in (".", "") else 0.0
                out.append((f[0], int(f[1]), int(f[2]), strand, score))
    except Exception:
        pass
    return out


def resolve_bed(rec):
    """Resolve a manifest record's BED path PORTABLY: the stored path if it exists, else
    ``config.DATA/'eclip'/<basename>``. Handles Windows-absolute manifest paths on any OS, so the
    shared dataset works unchanged on the laptop, the 5090 node, or a teammate's clone."""
    p = rec.get("path", "")
    if p and Path(p).exists():
        return Path(p)
    name = str(p).replace("\\", "/").rsplit("/", 1)[-1]
    return config.DATA / "eclip" / name


def rbp_peaks(rbp, manifest=None, max_peaks=100_000):
    manifest = manifest or _load_manifest()
    peaks = []
    for rec in manifest.get(rbp, []):
        peaks += read_bed(resolve_bed(rec), max_peaks)
    return peaks


def rbp_elements(rbp, manifest=None, lwin=600, max_n=None, top_by_score=True, rng=None):
    """ParnetDataElements for an RBP's peak windows (positives). outputs['eCLIP'] marks the center."""
    peaks = rbp_peaks(rbp, manifest)
    if not peaks:
        return []
    if top_by_score:
        peaks.sort(key=lambda p: -p[4])
    if max_n:
        peaks = peaks[:max_n] if top_by_score else [peaks[i] for i in (rng or np.random.default_rng(0)).choice(
            len(peaks), min(max_n, len(peaks)), replace=False)]
    import torch
    els = []
    for chrom, s, e, strand, score in peaks:
        seq = genome.centered_window(chrom, s, e, strand, lwin)
        if seq is None:
            continue
        sig = torch.zeros(1, lwin); sig[0, lwin // 2] = 1.0
        els.append(contract.make_element(seq, {"eCLIP": sig},
                                         name=f"{chrom}:{s}-{e}:{strand}", rbp=rbp, score=score))
    return els


def corpus(rbps, manifest=None, lwin=600, per_rbp=20, rng=None):
    """Mixed real-binding-window corpus across RBPs (for mix_coeff / interpretability)."""
    rng = rng or np.random.default_rng(0)
    manifest = manifest or _load_manifest()
    seqs = []
    for g in rbps:
        els = rbp_elements(g, manifest, lwin=lwin, max_n=per_rbp, top_by_score=True)
        seqs += [el["inputs"]["sequence"] for el in els]
    return seqs


def n2_windows(rbp, others, manifest=None, lwin=600, k=200, rng=None):
    """N2 negatives for `rbp`: peak windows of OTHER RBPs (protein-discriminative axis)."""
    rng = rng or np.random.default_rng(0)
    manifest = manifest or _load_manifest()
    pool = [g for g in others if g != rbp]
    seqs = []
    while len(seqs) < k and pool:
        g = pool[rng.integers(len(pool))]
        els = rbp_elements(g, manifest, lwin=lwin, max_n=3, top_by_score=False, rng=rng)
        seqs += [el["inputs"]["sequence"] for el in els]
    return seqs[:k]
