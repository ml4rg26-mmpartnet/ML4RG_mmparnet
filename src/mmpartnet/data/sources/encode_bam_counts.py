"""STUB: established single-nt 5' crosslink COUNTS from ENCODE eCLIP alignment BAMs.

This is the target the demo actually trains on (counts at the crosslink site, ~1 nt upstream of read2
5' end; Van Nostrand 2016). It is the established version of what encode_bigwig approximates with RPM
read-density. Fill `observed` from the GRCh38 BAMs (adapters.eclip_counts already resolves + reads them);
keep `windows`/`sequence` identical to encode_bigwig (same peaks, same hg38). Set cfg.target="counts".

To populate: copy encode_bigwig's rbps/windows/sequence, and back `observed` with
adapters.eclip_counts.readers_for(rec["exp"], bam_dir).profile(...). Lab-gated only by BAM availability
(public on ENCODE; large) — see docs/DATA_INVENTORY.md row "5' crosslink counts".
"""
from __future__ import annotations
from typing import Iterable, Optional

from ..base import DataSource, Window
from ..registry import register


@register("encode_bam_counts")
class EncodeBamCountsSource(DataSource):
    target_kind = "counts"

    def rbps(self) -> list:
        raise NotImplementedError(
            "encode_bam_counts: established 5'-crosslink-count source not filled yet. "
            "Reuse encode_bigwig's rbps/windows/sequence; back observed() with adapters.eclip_counts.")

    def windows(self, rbp: str) -> Iterable[Window]:
        raise NotImplementedError("encode_bam_counts.windows: reuse encode_bigwig.windows (same peaks).")

    def sequence(self, w: Window) -> Optional[str]:
        raise NotImplementedError("encode_bam_counts.sequence: reuse io.genome.window_seq (same hg38).")

    def observed(self, rbp: str, w: Window):
        raise NotImplementedError(
            "encode_bam_counts.observed: read single-nt 5' crosslink counts via "
            "adapters.eclip_counts.readers_for(rec['exp'], bam_dir).profile(chrom,start,end,strand).")
