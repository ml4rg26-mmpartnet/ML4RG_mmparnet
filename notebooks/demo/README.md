# notebooks/demo - runnable demonstration set

Four self-contained notebooks that (a) demonstrate the modular pipeline and (b) capture the tests that
verify our public-data proxies behave as intended. Each opens with **what is being tested / why / the
reasoning** plus the **math definitions**, runs a thin call into
`src/mmpartnet`, and closes with a results-and-interpretation cell whose numbers are pulled live (never
hardcoded), read beside the established nulls.

| notebook | what it demonstrates / tests | needs |
|---|---|---|
| `00_mmpartnet_demo.ipynb` | the modular data layer (`mmpartnet.data`) -> frozen PARNET -> profile recovery; how to switch source/format | PARNET weights + network |
| `01_proxy_validity_controls.ipynb` | density proxy vs established nulls (center-confounded, live) **and** the established 5' crosslink-counts target beating nulls (committed full-BAM run) | PARNET weights + network |
| `02_finetune_negative_controls.ipynb` | head-finetune vs pretrained, with Control 3 (random body) + Control 2 (faithful RBPNet objective): is the gain real transfer? | PARNET weights + network |
| `03_interpretability_mixcoeff.ipynb` | PARNET's per-RBP additive mix-coefficient (sequence- vs bias-driven) | committed JSON only |

`executed/` holds the run-with-outputs copies (00-02 from a full RTX 5090 run, 03 from the committed JSON).
Notebook 03 only needs the committed result JSONs; 00-02 need the PARNET weights + network.

## Run

Execute the whole set (after `scripts/fetch_all.sh` + the PARNET weights for 00-02):
```bash
bash scripts/run_demos.sh                 # -> notebooks/demo/executed/*_executed.ipynb
```

Or one notebook by hand:
```bash
MMP_GROUP=QKI,PTBP1 MMP_NWIN=6 \
  python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 \
  --output-dir /tmp/out notebooks/demo/00_mmpartnet_demo.ipynb
```

Knobs (env): `MMP_SOURCE` (data source), `MMP_GROUP` (RBP group), `MMP_CELL`, `MMP_NWIN`, `MMP_EPOCHS`;
notebook 02 uses `MMP_FT_GROUP` / `MMP_FT_NWIN` for its small finetune panel. All data paths are
environment-overridable (`ML4RG_DATA`, `ML4RG_HG38`, `ML4RG_REFS`, `ML4RG_PARNET_WEIGHTS`); see QUICKSTART.md.
