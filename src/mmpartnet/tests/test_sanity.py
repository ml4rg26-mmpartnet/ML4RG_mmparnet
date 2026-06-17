"""Phase-0 sanity tests — lock the foundation. Runnable as a script or via pytest.
  python -m mmpartnet.tests.test_sanity      (from poc/ml4rg_parnet/)
"""
from __future__ import annotations
import sys
from pathlib import Path

# allow standalone execution (add poc/ml4rg_parnet to path)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from mmpartnet import contract
from mmpartnet.io import cohort, embeddings
from mmpartnet.process.onehot import batch_onehot


def test_contract_roundtrip():
    t = torch.zeros(9, 600); t[0, 42] = 5; t[3, 100] = 2
    sp = contract.torch_dense_to_sparse(t)
    back = contract.torch_sparse_to_dense(sp)
    assert torch.equal(back, t), "dense->sparse->dense must be identity"
    el = contract.make_element("ACGTACGT", {"eCLIP": t}, name="chr1:1-8:+", rbp="QKI")
    assert contract.validate_element(el)
    print(f"  contract roundtrip OK (lab_source={contract.using_lab_source()})")


def test_cohort():
    pwms = cohort.load_pwms(); recs = cohort.parse_db()
    assert len(pwms) > 100 and len(recs) > 1000
    fam = cohort.attract_families(["QKI", "PTBP1", "PUM1"])
    assert fam.get("QKI") and cohort.norm_family(fam["QKI"][0])
    print(f"  cohort OK: pwms={len(pwms)} recs={len(recs)} QKI_family={fam['QKI'][0]}")


def test_embeddings():
    pr = embeddings.ProteinRep()
    n = len(pr.genes())
    assert n > 50, f"expected many pooled embeddings, got {n}"
    q = pr.esm("QKI")
    assert q is not None and q.shape[0] in (640, 1280)
    print(f"  embeddings OK: {n} RBPs, ESM dim={q.shape[0]}, QKI+STRING={pr.esm_string('QKI') is not None}")


def test_parnet_qki_sanity():
    """The locked CI gate: planted QKI motif -> QKI track profile peaks at the motif."""
    from mmpartnet.models.parnet import load_parnet
    m = load_parnet()
    motif = "TACTAACTACTAAC"; L = 1000; c = L // 2
    seq = "A" * ((L - len(motif)) // 2) + motif + "A" * (L - len(motif) - (L - len(motif)) // 2)
    x = batch_onehot([seq], device=m.device)
    out = m.full(x)
    assert set(["target", "control", "total", "mix_coeff"]) <= set(out)
    qi = m.track_index("QKI", "HepG2")
    prof = out["target"][0, qi].detach().cpu()
    argmax = int(prof.argmax())
    frac = float(prof[c - 20:c + 20].sum() / prof.sum())
    assert abs(argmax - c) < 30, f"QKI profile argmax {argmax} far from motif center {c}"
    assert frac > 0.25, f"QKI profile mass in motif window too low: {frac:.2f}"
    print(f"  PARNET QKI sanity OK: argmax={argmax} (center {c}), motif-window mass={frac:.2f}, "
          f"target shape={tuple(out['target'].shape)}")


if __name__ == "__main__":
    for fn in [test_contract_roundtrip, test_cohort, test_embeddings, test_parnet_qki_sanity]:
        print(f"[{fn.__name__}]", flush=True)
        fn()
    print("\nALL PHASE-0 SANITY TESTS PASSED.", flush=True)
