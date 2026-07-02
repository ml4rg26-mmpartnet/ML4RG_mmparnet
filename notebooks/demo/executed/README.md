# notebooks/demo/executed - executed copies (with outputs)

00-02 were executed on an a CUDA GPU GPU node on real public data (ENCODE eCLIP peaks + remote bigWig signal +
hg38 + frozen PARNET); 03 runs anywhere (committed JSON only). These are the demonstration record; the clean
(output-stripped) sources are one level up in `notebooks/demo/`.

Headline results:

| notebook | result |
|---|---|
| `00_mmpartnet_demo` | 176 windows (spliceosome-HepG2), mean profile Pearson +0.211 (density proxy, center-confounded - honest) |
| `01_proxy_validity_controls` | density +0.211 (center-bump +0.617 -> confounded) vs established 5' crosslink **counts +0.289 beats nulls = REAL shape** |
| `02_finetune_negative_controls` | finetune +0.206 -> +0.295 (delta +0.089); Control 3 random body -0.043 (=> real transfer); Control 2 faithful RBPNet objective +0.012 (=> gain is loss-sensitive) |
| `03_interpretability_mixcoeff` | 223 RBP-cell tracks, mix-coefficient median 0.78 (mostly sequence-driven, bias-dominated tail) |

Plots: `demo_profile.png` (00, best predicted-vs-observed window), `mixcoeff_hist.png` (03, mix-coefficient distribution).
