"""POPULATED demo source: public ENCODE eCLIP, read remotely (no multi-GB download).

What it wires (all already built; this just adapts them to the DataSource interface):
  - RBP -> binding windows:  peak BEDs via adapters.peaks (resolve_bed + read_bed), peak-centered to lwin
  - window -> sequence:      hg38 via io.genome.window_seq (revcomp on '-')
  - window -> observed:      per-nt RPM read-density via adapters.eclip_signal.SignalReader (HTTP range)

target_kind = "density": the ENCODE 'signal of unique reads' bigWig is RPM read-density, a SURROGATE for
the established single-nt 5' crosslink counts (see docs/DATA_INVENTORY.md). Swap to the established target
with `cfg.target="counts"` once the BAM-counts source (encode_bam_counts) is filled; the loader + model do
not change. Group selection (`cfg.group`) resolves through io.groups (curated set / family / SYM,SYM,...).
"""
from __future__ import annotations
import json
from functools import lru_cache
from typing import Iterable, Optional

from ... import config
from ...io import genome, groups
from ...adapters import peaks as peaks_adapter
from ...adapters import eclip_signal
from ..base import DataSource, Window
from ..registry import register

#: over-sample factor: read this many peaks so the >=min_sum read filter still yields ~nwin windows
_OVERSAMPLE = 4


@register("encode_bigwig")
class EncodeBigwigSource(DataSource):
    target_kind = "density"

    @lru_cache(maxsize=1)
    def _manifest(self) -> dict:
        return json.loads((config.DATA / "eclip_manifest.json").read_text())

    def _rec(self, rbp: str) -> Optional[dict]:
        """The manifest record for this RBP at cfg.cell (peak BED path + ENCODE experiment accession)."""
        return next((r for r in self._manifest().get(rbp, []) if r.get("cell") == self.cfg.cell), None)

    def rbps(self) -> list:
        man = self._manifest()
        have = [g for g in man if any(r.get("cell") == self.cfg.cell for r in man[g])]
        want = groups.resolve(self.cfg.group)        # [] => no filter (all)
        if not want:
            return sorted(have)
        avail = set(have)
        return [g for g in want if g in avail]       # preserve the group's order

    def windows(self, rbp: str) -> Iterable[Window]:
        rec = self._rec(rbp)
        if rec is None:
            return
        peaks = peaks_adapter.read_bed(peaks_adapter.resolve_bed(rec))
        peaks.sort(key=lambda p: -p[4])              # by narrowPeak score, like the demo
        lwin = self.cfg.lwin
        for chrom, s, e, strand, _score in peaks[: self.cfg.nwin * _OVERSAMPLE]:
            c = (s + e) // 2
            ws = c - lwin // 2
            yield Window(chrom=chrom, start=ws, end=ws + lwin, strand=strand, rbp=rbp)

    def sequence(self, w: Window) -> Optional[str]:
        return genome.window_seq(w.chrom, w.start, w.end, w.strand)

    @lru_cache(maxsize=64)
    def _reader(self, rbp: str):
        """Strand-aware bigWig reader for the RBP's ENCODE experiment (opened once, cached)."""
        rec = self._rec(rbp)
        if rec is None:
            return None
        u = eclip_signal.resolve_signal_urls(rec["exp"])
        return eclip_signal.SignalReader(u.get("plus"), u.get("minus"))

    def observed(self, rbp: str, w: Window):
        reader = self._reader(rbp)
        if reader is None or not reader.has(w.chrom):
            return None
        return reader.profile(w.chrom, w.start, w.end, w.strand)
