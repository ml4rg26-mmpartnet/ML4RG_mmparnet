# mmpartnet_out/ - precomputed result JSONs (committed)

Small committed JSONs so the demo notebooks render their numbers on a fresh clone without re-running the
(slow, asset-dependent) experiments. Notebooks read their headline numbers from these files
programmatically (never hardcoded).

| file | loaded by | headline |
|------|-----------|----------|
| `recover_demo_profile_counts.json` | `01` | established 5' crosslink-count target: mean Pearson +0.289, beats the established nulls |
| `mixcoeff_per_rbp.json` | `03` | PARNET additive mix-coefficient per RBP-cell track (median ~0.78) |

These were computed on the leaked all-223 PARNET + proxy RIBEX substrate; see the caveats in `../README.md`.
