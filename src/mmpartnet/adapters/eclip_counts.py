"""Control 1: the ESTABLISHED eCLIP target = single-nucleotide crosslink COUNTS from ENCODE BAMs.

The PARNET/RBPNet objective (Van Nostrand 2016, Nat. Methods 13:508; YeoLab/ENCODE eCLIP pipeline) is
defined on single-nt crosslink counts, NOT the RPM read-DENSITY that `eclip_signal` reads from the
bigWigs. The crosslink site is the genomic position **1 nt 5' (upstream) of the 5' end of read2** (the
reverse-transcription truncation read), on read2's mapping strand. This module re-derives that target
from the public GRCh38 alignment BAMs (resolved via the same ENCODE API path as `eclip_signal`) and
exposes the SAME interface as `SignalReader` (`.has`, `.profile`) so it is a drop-in `--target counts`.

STRAND CONVENTION (the one error-prone bit): ENCODE eCLIP is read2-forward — read2 maps to the RNA
(sense) strand — so a read2's crosslink is assigned to read2's mapping strand. `validate_strand()`
confirms this by checking the derived counts correlate POSITIVELY (not negatively) with the
strand-separated bigWig density; if a dataset is the opposite convention it is caught there.

BAMs are large (~0.3 GB each); download once to node-local scratch and open locally (remote pysam
range-reads need a co-located .bai and are fragile). `download_bams()` fetches + indexes.
"""
from __future__ import annotations
import json
import os
import urllib.request
from functools import lru_cache
import numpy as np

ENCODE = "https://www.encodeproject.org"


def _exp_json(acc):
    req = urllib.request.Request(f"{ENCODE}/experiments/{acc}/?format=json",
                                 headers={"Accept": "application/json", "User-Agent": "mmpartnet"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())


def _bam_hrefs(files, assembly):
    """GRCh38 alignment BAM hrefs (released). Skips hg19 + non-alignment outputs."""
    out = []
    for f in files:
        if (f.get("file_format") == "bam" and f.get("assembly") == assembly
                and "alignment" in (f.get("output_type") or "").lower()
                and (f.get("status") or "released") == "released"):
            out.append(ENCODE + f["href"])
    return out


@lru_cache(maxsize=128)
def resolve_bam_urls(exp_acc: str, assembly: str = "GRCh38") -> dict:
    """{'bams':[...], 'control_exp':acc, 'control_bams':[...]} — mirrors resolve_signal_urls."""
    e = _exp_json(exp_acc)
    bams = _bam_hrefs(e["files"], assembly)
    ctrl_acc = (e.get("possible_controls") or [{}])[0].get("accession")
    ctrl_bams = _bam_hrefs(_exp_json(ctrl_acc)["files"], assembly) if ctrl_acc else []
    return {"bams": bams, "control_exp": ctrl_acc, "control_bams": ctrl_bams}


def download_bams(hrefs, dest_dir):
    """Download each BAM to dest_dir (skip if present) + ensure a .bai. Returns local paths."""
    import pysam
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for h in hrefs:
        name = h.rsplit("/", 1)[-1]
        p = os.path.join(dest_dir, name)
        if not os.path.exists(p):
            urllib.request.urlretrieve(h, p)
        if not (os.path.exists(p + ".bai") or os.path.exists(p[:-4] + ".bai")):
            pysam.index(p)
        paths.append(p)
    return paths


class CountReader:
    """5'-crosslink-count profile over local eCLIP BAM(s). Same interface as SignalReader so it drops
    into the experiments. Sums crosslink counts across replicate BAMs; strand-aware; reversed on '-'
    to the 5'->3' orientation PARNET is fed (matching SignalReader.profile)."""

    def __init__(self, bam_paths, upstream=1):
        import pysam
        self._bams = [pysam.AlignmentFile(p, "rb") for p in bam_paths]
        self._chroms = set()
        for b in self._bams:
            self._chroms.update(b.references)
        self.upstream = upstream

    def has(self, chrom):
        return chrom in self._chroms

    def _xlink_pos_strand(self, read):
        """Crosslink genomic position + strand for the truncation read (read2 if paired, else the read).
        Crosslink = `upstream` nt 5' of the read's 5' end, on the read's mapping strand."""
        if read.is_paired and not read.is_read2:
            return None  # use read2 only (the RT-truncation read) for paired eCLIP
        if read.is_reverse:
            return read.reference_end - 1 + self.upstream, "-"   # 5' end at the high coord; 5' = +offset
        return read.reference_start - self.upstream, "+"          # 5' end at the low coord; 5' = -offset

    def profile(self, chrom, start, end, strand="+"):
        if chrom not in self._chroms:
            return None
        start = int(start); end = int(end); L = end - start
        if L <= 0:
            return None
        v = np.zeros(L, dtype=np.float64)
        try:
            for b in self._bams:
                # widen the fetch a little so reads whose 5'/crosslink falls in-window are caught
                for r in b.fetch(chrom, max(0, start - 5), end + 5):
                    if r.is_unmapped or r.is_duplicate or r.is_secondary or r.is_supplementary:
                        continue
                    ps = self._xlink_pos_strand(r)
                    if ps is None:
                        continue
                    pos, st = ps
                    if st != strand or pos < start or pos >= end:
                        continue
                    v[pos - start] += 1.0
        except Exception:
            return None
        return v[::-1].copy() if strand == "-" else v


def readers_for(exp_acc, dest_dir, assembly: str = "GRCh38"):
    """(eclip_count_reader, control_count_reader_or_None, urls) — counts analogue of
    eclip_signal.readers_for. Downloads the eCLIP + control BAMs into dest_dir."""
    u = resolve_bam_urls(exp_acc, assembly)
    ecl = CountReader(download_bams(u["bams"], dest_dir)) if u["bams"] else None
    ctrl = CountReader(download_bams(u["control_bams"], dest_dir)) if u["control_bams"] else None
    return ecl, ctrl, u


def validate_strand(exp_acc, manifest_rec, count_reader, signal_reader, n=20):
    """Sanity that the strand convention is right: the crosslink-count profile should correlate
    POSITIVELY with the established strand-separated bigWig density over the same windows. Returns the
    mean Pearson between counts and density; a strongly NEGATIVE value means the strand is flipped."""
    import gzip
    from . import peaks as peaks_adapter
    rs = []
    path = peaks_adapter.resolve_bed(manifest_rec)
    rows = []
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                strand = f[5] if len(f) > 5 and f[5] in "+-" else "+"
                rows.append((f[0], int(f[1]), int(f[2]), strand))
    for chrom, s, e, strand in rows[:n * 3]:
        if len(rs) >= n or not (count_reader.has(chrom) and signal_reader.has(chrom)):
            continue
        c = (s + e) // 2; ws = c - 300
        cnt = count_reader.profile(chrom, ws, ws + 600, strand)
        den = signal_reader.profile(chrom, ws, ws + 600, strand)
        if cnt is None or den is None or cnt.sum() < 5 or den.sum() < 5:
            continue
        x = cnt - cnt.mean(); y = den - den.mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        if d > 1e-9:
            rs.append(float((x * y).sum() / d))
    return float(np.mean(rs)) if rs else float("nan")
