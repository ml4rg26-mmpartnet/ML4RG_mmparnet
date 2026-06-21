#!/usr/bin/env python
"""Sanity-check the shared PARNET 7M checkpoint.

This verifies the baseline wiring before any training:
1. load the 223-track pretrained PARNET checkpoint;
2. confirm a known RBP-cell track exists;
3. plant a QKI motif in an artificial RNA window and check that QKI_HepG2 peaks nearby.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REFS = REPO.parent / "parnet_refs"
DEFAULT_WEIGHTS = Path(
    "/home/dgu/storage_ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt"
)


def main() -> None:
    os.environ.setdefault("ML4RG_REFS", str(DEFAULT_REFS))
    os.environ.setdefault("ML4RG_PARNET_WEIGHTS", str(DEFAULT_WEIGHTS))

    from mmpartnet.models.parnet import load_parnet
    from mmpartnet.process.onehot import batch_onehot

    model = load_parnet(device="cpu")
    qki_idx = model.track_index("QKI", "HepG2")

    motif = "TACTAACTACTAAC"
    length = 1000
    center = length // 2
    left = (length - len(motif)) // 2
    seq = "A" * left + motif + "A" * (length - len(motif) - left)

    x = batch_onehot([seq], device=model.device)
    out = model.full(x)
    profile = out["target"][0, qki_idx].detach().cpu()
    argmax = int(profile.argmax())
    center_mass = float(profile[center - 20 : center + 20].sum() / profile.sum())
    passed = abs(argmax - center) < 30 and center_mass > 0.25

    print(f"weights: {os.environ['ML4RG_PARNET_WEIGHTS']}")
    print(f"refs:    {os.environ['ML4RG_REFS']}")
    print(f"model:   {type(model.m)}")
    print(f"tracks:  {len(model.syms)}")
    print(f"QKI_HepG2 index: {qki_idx}")
    print(f"target shape:    {tuple(out['target'].shape)}")
    print(f"mix_coeff shape: {tuple(out['mix_coeff'].shape)}")
    print(f"motif center:    {center}")
    print(f"profile argmax:  {argmax}")
    print(f"center +/-20 mass: {center_mass:.4f}")
    print(f"PASS: {passed}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
