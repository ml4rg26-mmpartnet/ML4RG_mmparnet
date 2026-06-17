"""Cohort / motif / family parsing — the ONE canonical home for functions that were
duplicated across ~7 realdata scripts (best_rec x7, norm_family x2, load_pwms/parse_db
re-exported everywhere). Faithful to build_cohort.py + protein_entropy_geometry.attract_families
+ cross_species_specificity.attract_all, reading the same files via config paths.
"""
from __future__ import annotations
import json
from functools import lru_cache
import numpy as np

from .. import config

BASES = ["A", "C", "G", "U"]  # ATtRACT PWM column order A C G T/U


@lru_cache(maxsize=1)
def load_pwms() -> dict:
    """pwm.txt -> {matrix_id: ndarray(positions, 4)} (build_cohort.load_pwms)."""
    pwms, cur, rows = {}, None, []
    for line in config.PWM_TXT.read_text().splitlines():
        if line.startswith(">"):
            if cur is not None and rows:
                pwms[cur] = np.array(rows, dtype=np.float64)
            cur = line[1:].split("\t")[0].strip(); rows = []
        elif line.strip():
            rows.append([float(x) for x in line.split()])
    if cur is not None and rows:
        pwms[cur] = np.array(rows, dtype=np.float64)
    return pwms


@lru_cache(maxsize=1)
def parse_db() -> list:
    """ATtRACT_db.txt -> list of record dicts (build_cohort.parse_db)."""
    lines = config.ATTRACT_DB.read_text().splitlines()
    header = lines[0].split("\t")
    idx = {name: i for i, name in enumerate(header)}
    recs = []
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) < len(header):
            continue
        recs.append({
            "gene": f[idx["Gene_name"]], "organism": f[idx["Organism"]],
            "motif": f[idx["Motif"]], "family": f[idx["Family"]],
            "matrix_id": f[header.index("Matrix_id")], "score": f[-1],
            "database": f[idx["Database"]],
        })
    return recs


def qnum(score: str) -> float:
    return float(str(score).replace("*", "").strip() or 0.0)


def best_rec(cands: list, pwms: dict | None = None) -> dict:
    """Canonical best-record pick (was duplicated 7x): prefer motif width 4-12, then Qscore,
    then curated db. `cands` = records for one (gene[, organism])."""
    pwms = pwms or load_pwms()
    return max(cands, key=lambda r: (
        1 if 4 <= pwms[r["matrix_id"]].shape[0] <= 12 else 0,
        qnum(r["score"]),
        1 if r["database"] in ("C", "R", "S", "AEDB") else 0))


def norm_family(s: str) -> str:
    """Normalize an ATtRACT Family string to canonical, sorted, de-duped form (was duplicated 2x)."""
    return ";".join(sorted(frozenset(s.replace(" ", "").split(";"))))


def attract_families(genes, organism="Homo_sapiens"):
    """Per gene (one organism), best_rec -> (normalized family, pwm). Faithful to
    protein_entropy_geometry.attract_families. Returns {gene: (family, pwm)}."""
    pwms = load_pwms(); recs = parse_db()
    byg = {}
    for r in recs:
        if r["organism"] == organism and r["matrix_id"] in pwms:
            byg.setdefault(r["gene"], []).append(r)
    out = {}
    for g in genes:
        if g in byg:
            b = best_rec(byg[g], pwms)
            out[g] = (norm_family(b["family"]), pwms[b["matrix_id"]])
    return out


def attract_all(genes_with_species=None):
    """best_rec per (GENE, organism) over ALL organisms -> {(GENE,Org): (family, pwm)}
    (faithful to cross_species_specificity.attract_all). Used for ortholog work."""
    pwms = load_pwms()
    byk = {}
    for r in parse_db():
        if r["matrix_id"] not in pwms:
            continue
        byk.setdefault((r["gene"].upper(), r["organism"]), []).append(r)
    return {k: (norm_family(best_rec(c, pwms)["family"]), pwms[best_rec(c, pwms)["matrix_id"]])
            for k, c in byk.items()}


@lru_cache(maxsize=1)
def load_cohort_json() -> dict:
    """The prebuilt ENCORE x ATtRACT cohort (45 RBPs): {gene: {family, pwm, motif, ...}}."""
    return json.loads(config.COHORT_JSON.read_text())
