"""Build the per-nucleotide eCLIP PROFILE target from public ENCODE signal bigWigs.

This is the GENUINE eCLIP signal (same ENCODE experiments as our peaks) — the per-position data
the PARNET demo's profile objective needs and that peak BEDs cannot provide. We resolve the GRCh38
'plus/minus strand signal of unique reads' bigWigs (+ the paired SMInput control) via the ENCODE
API, and read per-window profiles REMOTELY over HTTP range requests (no multi-GB download).

Backend: pybigtools (Windows-installable). Strand handling: the '-'-strand profile is reversed so
it is in transcript 5'->3' orientation, matching the revcomp sequence PARNET is fed.
"""
from __future__ import annotations
import json
import urllib.request
from functools import lru_cache
import numpy as np

ENCODE = "https://www.encodeproject.org"


def _exp_json(acc):
    req = urllib.request.Request(f"{ENCODE}/experiments/{acc}/?format=json",
                                 headers={"Accept": "application/json", "User-Agent": "mmpartnet"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())


def _strand_urls(files, assembly):
    out = {}
    for f in files:
        if f.get("file_format") == "bigWig" and f.get("assembly") == assembly:
            ot = (f.get("output_type") or "").lower()
            if "plus strand signal" in ot:
                out.setdefault("plus", ENCODE + f["href"])
            elif "minus strand signal" in ot:
                out.setdefault("minus", ENCODE + f["href"])
    return out


@lru_cache(maxsize=128)
def resolve_signal_urls(exp_acc: str, assembly: str = "GRCh38") -> dict:
    """{'plus','minus','control_exp','control':{'plus','minus'}} for an ENCODE eCLIP experiment."""
    e = _exp_json(exp_acc)
    sig = _strand_urls(e["files"], assembly)
    ctrl_acc = (e.get("possible_controls") or [{}])[0].get("accession")
    ctrl = _strand_urls(_exp_json(ctrl_acc)["files"], assembly) if ctrl_acc else {}
    return {"plus": sig.get("plus"), "minus": sig.get("minus"),
            "control_exp": ctrl_acc, "control": ctrl}


def readers_for(exp_acc: str, assembly: str = "GRCh38"):
    """(eclip_reader, control_reader_or_None, urls) for an ENCODE eCLIP experiment — wires BOTH the
    eCLIP signal and the paired SMInput **control** track (row 4 of the data inventory)."""
    u = resolve_signal_urls(exp_acc, assembly)
    eclip = SignalReader(u["plus"], u["minus"])
    cu = u.get("control") or {}
    control = SignalReader(cu.get("plus"), cu.get("minus")) if (cu.get("plus") or cu.get("minus")) else None
    return eclip, control, u


class SignalReader:
    """Open the plus/minus bigWigs once; read strand-aware per-window profiles (remote or local)."""

    def __init__(self, plus_url, minus_url):
        import pybigtools
        self.plus = pybigtools.open(plus_url) if plus_url else None
        self.minus = pybigtools.open(minus_url) if minus_url else None
        self._chroms = set(self.plus.chroms()) if self.plus else set()

    def has(self, chrom):
        return chrom in self._chroms

    def profile(self, chrom, start, end, strand="+"):
        """Per-nt |signal| over [start,end), 5'->3' (reversed on '-'). None on error/out-of-bounds."""
        bw = self.minus if strand == "-" else self.plus
        if bw is None or chrom not in self._chroms:
            return None
        try:
            v = np.asarray(bw.values(chrom, int(start), int(end), fillna=0.0), dtype=np.float64)
        except Exception:
            return None
        v = np.abs(np.nan_to_num(v))
        if v.shape[0] != int(end) - int(start):
            return None
        return v[::-1].copy() if strand == "-" else v
