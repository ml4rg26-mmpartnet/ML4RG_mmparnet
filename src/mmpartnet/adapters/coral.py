"""Adapter: our eCLIP / affinity data -> CORAL's CSV schema, so we can run OUR data through CORAL's
exact pipeline (RNA-protein INTERACTION detection with random-repair negatives) and compare on the
same footing.

CORAL train.csv schema:  RNA_id, Protein_id, labels, RNA_seqs, Prot_seqs

The full window/transcript builders (they need the eCLIP dataset + GENCODE) live in
`scripts/eclip_to_coral.py` and `scripts/eclip_to_coral_transcript.py`. This module gives the importable
WRITER + a round-trip VALIDATOR so a teammate can trust the conversion without re-reading the scripts.
"""
from __future__ import annotations

import csv

CORAL_COLUMNS = ["RNA_id", "Protein_id", "labels", "RNA_seqs", "Prot_seqs"]


def write_coral_csv(rows, out) -> int:
    """rows: iterable of (RNA_id, Protein_id, label, RNA_seq, Prot_seq). Writes a CORAL-schema CSV;
    returns the number of data rows written."""
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CORAL_COLUMNS)
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def validate_roundtrip(csv_path, require_balanced=False) -> dict:
    """Check a CORAL CSV: header matches, labels in {0,1}, no empty sequences; report pos/neg + unique
    RNA/protein counts. Raises AssertionError on a schema violation (so a bad conversion fails loudly)."""
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == CORAL_COLUMNS, f"bad header {header} != {CORAL_COLUMNS}"
        npos = nneg = bad = 0
        rna, prot = set(), set()
        for row in r:
            if len(row) != 5:
                bad += 1
                continue
            rid, pid, lab, rseq, pseq = row
            assert lab in ("0", "1"), f"label not binary: {lab!r}"
            if not rseq or not pseq:
                bad += 1
                continue
            npos += (lab == "1")
            nneg += (lab == "0")
            rna.add(rid)
            prot.add(pid)
    assert bad == 0, f"{bad} malformed/empty rows"
    out = {"pos": int(npos), "neg": int(nneg), "n_rna": len(rna), "n_prot": len(prot),
           "total": int(npos + nneg)}
    if require_balanced:
        assert abs(npos - nneg) <= 0.05 * (npos + nneg + 1), f"unbalanced: {npos} pos vs {nneg} neg"
    return out
