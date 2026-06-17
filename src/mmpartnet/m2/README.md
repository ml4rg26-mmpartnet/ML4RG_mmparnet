# `mmpartnet.m2` — Milestone-2 scaffold (mostly stubs + contracts)

M2 = an M1 FiLM-conditioned head trained under `splits.rbp_holdout` on a
leave-out-pretrained PARNET, evaluated zero-shot with the two mandatory controls.

| piece | status | role |
|-------|--------|------|
| `leaveout_parnet.load_leaveout_parnet(held_rbps)` | seam + guard | swap-in #1; refuses to pass off all-223 weights as leave-out (lab-gated checkpoint) |
| `generalization.zero_shot_eval(...)` | stub | held-RBP N2 auROC + protein-shuffle + RNA-matched-N2 controls; `honest` flag |

Never report an M2 number without both standing controls beside it (CONTRACT.md). The contracts are
fixed now; bodies fill on M2 data. Decisive read: family-holdout permutation gap stays positive on the
leave-out PARNET -> protein carries transferable cross-family code; collapses -> the all-223 gap was leakage.
