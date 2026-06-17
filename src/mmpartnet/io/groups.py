"""Named RBP groups — the parsed-option 'which data group' selector for the scripts + experiments.

A 'group' = a subset of RBP symbols. The scripts (e.g. ``fetch_all.sh`` / ``fetch_data.sh``) take
``--group <name>`` and export ``ML4RG_GROUP``; experiments call ``groups.select(rbps)`` to scope
their RBP list. Unset / 'all' = no filter. Groups resolve from the REAL metadata
(``encode_rbps.json`` + ``cohort.json`` families) plus a few curated functional sets, so 'an initial
run on a small credible slice' is one flag.

Resolution order for a name: curated set -> 'cohort' (the 45 ATtRACT-joined) -> an ATtRACT family
-> a comma-list of explicit symbols -> a single symbol.
"""
from __future__ import annotations
import os
import json

from .. import config

# Curated functional sets (symbols intersected with what's actually available locally).
CURATED = {
    "spliceosome": ["SF3B4", "SF3A3", "BUD13", "AQR", "PRPF8", "EFTUD2", "SMNDC1",
                    "U2AF1", "U2AF2", "RBM22", "XRN2"],
    "igf2bp": ["IGF2BP1", "IGF2BP2", "IGF2BP3"],
    "demo6": ["QKI", "PTBP1", "IGF2BP1", "SRSF1", "HNRNPC", "TARDBP"],
}


def _families() -> dict:
    """{family_lower: [rbps]} from cohort.json (the ATtRACT-joined cohort)."""
    p = config.COHORT_JSON
    fams: dict = {}
    if p.exists():
        for g, rec in json.loads(p.read_text()).items():
            fam = (rec.get("family") or "?").split(";")[0].strip().lower()
            if fam and fam != "?":
                fams.setdefault(fam, []).append(g)
    return fams


def resolve(name: str | None) -> list:
    """name -> list of RBP symbols. '' / None / 'all' -> [] (meaning: no filter)."""
    if not name or name.lower() == "all":
        return []
    n = name.lower()
    if n in CURATED:
        return list(CURATED[n])
    if n == "cohort":
        return list(json.loads(config.COHORT_JSON.read_text())) if config.COHORT_JSON.exists() else []
    fams = _families()
    if n in fams:
        return fams[n]
    if "," in name:
        return [s.strip() for s in name.split(",") if s.strip()]
    return [name]


def select(rbps, name: str | None = None) -> list:
    """Filter ``rbps`` to the active group (``ML4RG_GROUP`` env unless ``name`` given), preserving
    order. Empty/unknown group -> unchanged. A group that intersects to nothing -> unchanged (we
    never silently return an empty run); the caller's log shows the resulting count."""
    name = name if name is not None else os.environ.get("ML4RG_GROUP", "")
    want = resolve(name)
    if not want:
        return list(rbps)
    want_set = {w.upper() for w in want}
    sub = [g for g in rbps if g.upper() in want_set]
    return sub or list(rbps)


def available() -> dict:
    fams = _families()
    n_all = len(json.loads(config.ENCODE_RBPS.read_text())) if getattr(config, "ENCODE_RBPS", None) \
        and config.ENCODE_RBPS.exists() else None
    return {"curated": sorted(CURATED), "cohort": 45, "families": sorted(fams), "all": n_all}


def list_str() -> str:
    a = available()
    fams = a["families"]
    return ("groups: all | cohort | " + " | ".join(a["curated"]) + " | <family> | <SYM1,SYM2,...>\n"
            f"  available families ({len(fams)}): " + ", ".join(fams[:24]) +
            (" ..." if len(fams) > 24 else ""))
