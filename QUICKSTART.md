# MultiModal PARNET - quickstart (clone, fetch, run)

ML4RG SS26 Project 07. This gets you from a fresh clone to running the demo notebooks on a CUDA GPU node
(an RTX 5090, or any CUDA machine). The pipeline conditions the lab's frozen PARNET (RNA -> per-nt eCLIP profile) on a
protein representation; this repo is the runnable substrate + the tests that verify our public-data proxies.

## 1. Environment

```bash
git clone <this-repo> mmpartnet && cd mmpartnet
python3 -m venv .venv && . .venv/bin/activate     # Python 3.12 (e.g. on a GPU node)
pip install --upgrade pip
# Blackwell (RTX 5090, sm_120) needs the cu128 nightly torch:
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -e ".[eclip,viz,interp]"              # pulls jupyter/nbconvert/ipykernel via [viz]
pip install gin-config tqdm                        # PARNET unpickler deps
```

## 2. Fetch the data (public)

```bash
bash scripts/fetch_all.sh                 # all RBP peak BEDs + hg38 (~3GB) into ./data/
# or a smaller slice:  bash scripts/fetch_all.sh --group spliceosome
# if hg38 is already on the node:  ML4RG_HG38=/path/hg38.fa bash scripts/fetch_all.sh --no-hg38
```

This pulls public ENCODE eCLIP peak BEDs (by accession) and hg38. The per-nucleotide eCLIP **signal** is
read remotely at run time (HTTP range requests), so it is not downloaded.

## 3. PARNET weights (gated - ask a supervisor)

Not public. Place them where the config expects (or set `ML4RG_PARNET_WEIGHTS`):

```
data/refs/parnet/models/NewRBPNet_7M_Penalty-0.0_20250107.pt
data/refs/parnet/parnet/assets/ENCODE.idx2symbol-cell.pt
```

Without weights you can still run notebook **03** (committed result JSONs); **00-02** need the weights + network.

## 4. Run

```bash
bash scripts/run_demos.sh                 # executes all demo notebooks -> notebooks/demo/executed/
```

or open `notebooks/demo/*.ipynb` in Jupyter. Each notebook states what/why/reasoning + the math, runs a thin
call into `src/mmpartnet`, and pulls its numbers live. Pre-run copies (with outputs) are in
`notebooks/demo/executed/` so you can read the results before running anything.

## What is here

| path | what |
|---|---|
| `notebooks/demo/` | 4 runnable demo notebooks (00 data layer, 01 proxy validity, 02 finetune controls, 03 interpretability) + `executed/` |
| `src/mmpartnet/` | the package: `data/` `protein/` `splits/` `m2/` swappable layers + `models/`, `experiments/`, `io/`, `adapters/` |
| `docs/` | `DATA_INVENTORY.md` (have/surrogate/missing) + diagrams; `CONTRACT.md` is at the repo root |
| `scripts/` | `fetch_all.sh`, `run_demos.sh`, `fetch_data.sh`, `build_embeddings.py` |
| `metadata/` | public ENCODE metadata (eclip manifest, cohort, RBP list) used by the fetch |

The package is built around one data contract + four one-line config swap-ins (PARNET weights, protein rep,
data substrate, split axis); see `CONTRACT.md`. Everything path-related is env-overridable, so a fresh clone
runs zero-edit once the data is fetched.
