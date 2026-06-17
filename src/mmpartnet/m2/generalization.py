"""M2 zero-shot cross-RBP generalization eval (stub + contract).

The M2 question: does conditioning on the protein rep let the model predict binding for an RBP with
NO eCLIP data (held out via splits.rbp_holdout)? This wires the existing building blocks; it is a stub
so the team fills the body against M2 data, but the contract + the MANDATORY controls are fixed here.

Building blocks present here: data.iter_records (windows+seq+target), protein.get_protein (rep),
models.parnet body feats. The team adds a FiLM-conditioned head + the two mandatory controls
(protein-permutation gate + RNA-matched hard-N2) when filling this stub.
"""
from __future__ import annotations


def zero_shot_eval(*, train_source, test_source, protein, held_rbps, split=None, device="cpu"):
    """Train a FiLM-conditioned head on `train_source` RBPs, evaluate on `held_rbps` (zero-shot).

    Contract (return): {
      "held_rbps": [...],
      "n2_auroc_heldout": float,                 # mean per-held-RBP N2 auROC
      "control_protein_shuffle": float,          # MUST collapse toward 0.5 (else protein bypass)
      "control_rna_matched_n2": float,           # RNA-matched hard-N2 (guards the RNA shortcut)
      "honest": bool,                            # True only if controls collapse AND PARNET is leave-out
    }
    NEVER report n2_auroc_heldout without both controls beside it (CONTRACT.md). honest=False unless
    the body was a leave-out-pretrained PARNET (m2.leaveout_parnet) for the held RBPs.
    """
    raise NotImplementedError(
        "m2.zero_shot_eval: fill the FiLM-conditioned head + the protein-permutation and "
        "RNA-matched-N2 controls. Requires (a) splits.rbp_holdout held_rbps, and (b) a "
        "leave-out-pretrained PARNET (m2.leaveout_parnet) for an honest zero-shot claim.")
