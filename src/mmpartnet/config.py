"""Central paths + run configuration. Everything else reads from here so swapping
substrate / cohort / loss / model is a CONFIG change, not a code change
(Moyon's steer: less architecture, more loss + signal).

Portability contract: every path defaults to a REPO-RELATIVE location and is
overridable by an environment variable, so a fresh `git clone` + `scripts/fetch_data.sh`
runs zero-edit on any machine (laptop, an HPC node, CI). No absolute `D:/...` literals.

The four gated swap-in points (see CONTRACT.md) are all one-line config changes:
  1. PARNET weights  -> ML4RG_PARNET_WEIGHTS  (leaked all-223 -> a leave-out-pretraining)
  2. protein rep     -> RunConfig.protein     ("esm650_pooled"/"ribex_proxy" -> "ribex_real")
  3. data substrate  -> RunConfig.substrate   ("peaks" public ENCODE -> "hfds" encode.filtered)
  4. split axis      -> RunConfig.split        ("naive"/"family" -> "rbp_holdout")
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Repo root = three parents up from this file (src/mmpartnet/config.py) ──────
_REPO = Path(__file__).resolve().parents[2]

# ── Data + results roots (env-overridable, repo-relative defaults) ────────────
DATA = Path(os.environ.get("ML4RG_DATA", _REPO / "data"))            # gitignored; fetch_data.sh populates
REALDATA = Path(os.environ.get("ML4RG_REALDATA", _REPO))            # parent of the output dir
RESULTS = Path(os.environ.get("ML4RG_RESULTS", _REPO / "mmpartnet_out"))  # precomputed JSONs (committed, small)

# ── External reference assets (lab PARNET source pkg + weights) ───────────────
_VM_REFS = Path("/home/dgu/workspace/parnet_refs")
_DEFAULT_REFS = _VM_REFS if _VM_REFS.exists() else DATA / "refs"
REFS = Path(os.environ.get("ML4RG_REFS", _DEFAULT_REFS))
PARNET_PKG = REFS / "parnet" / "parnet"                              # the parnet source package
_VM_PARNET_WEIGHTS = Path("/mnt/storage1/ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt")
_DEFAULT_PARNET_WEIGHTS = (
    _VM_PARNET_WEIGHTS
    if _VM_PARNET_WEIGHTS.exists()
    else REFS / "parnet" / "models" / "NewRBPNet_7M_Penalty-0.0_20250107.pt"
)
PARNET_WEIGHTS = Path(os.environ.get(
    "ML4RG_PARNET_WEIGHTS", _DEFAULT_PARNET_WEIGHTS))
PARNET_IDX2SYM = PARNET_PKG / "assets" / "ENCODE.idx2symbol-cell.pt"
LAB_UTILS_SRC = REFS / "parnet--demo--train-models" / "src"         # parnet_demo_utils (the contract)
SCAFFOLD = REFS / "parnet--demo--train-models"

# ── Genome + embedding / motif assets (under DATA, gitignored) ────────────────
HG38 = Path(os.environ.get("ML4RG_HG38", DATA / "hg38.fa"))
EMB_POOLED = DATA / "embeddings_all.npz"          # ESM2-650M pooled (640-d), human
EMB_XSPECIES = DATA / "embeddings_xspecies.npz"   # 132 orthologs
EMB_PERRES = DATA / "perres64.npz"
PE_STRING = DATA / "pe_string.npz"                # STRING-PPI personalized-PageRank PE
ATTRACT_DB = DATA / "ATtRACT_db.txt"
PWM_TXT = DATA / "pwm.txt"
COHORT_JSON = DATA / "cohort.json"
ENCODE_RBPS = DATA / "encode_rbps.json"   # list[str] of the ENCODE eCLIP RBP symbols (group source)


@dataclass
class RunConfig:
    """A single experiment = one RunConfig. Phases/threads read these flags.
    The gated swap-ins (substrate/protein + the PARNET weight env var) flip here."""
    substrate: str = "peaks"          # "peaks" (public ENCODE eCLIP, now) | "hfds" (lab canonical)
    cohort_filter: str = "all"        # "all" | "rrm" | "spliceosome" | <family>
    split: str = "naive"              # "naive" | "family" | "rbp_holdout"
    protein: str = "esm650_pooled"    # "esm650_pooled" | "ribex_proxy" | "ribex_real"
    loss: str = "bidir_n2"            # bidirectional symmetric N2 contrast WINS (3-seed); also bce/infonce/margin
    model: str = "parnet_7m"          # frozen-body PARNET checkpoint id
    lwin: int = 600                   # window length (matches PARNET training tiles)
    seeds: tuple = (0, 1, 2)
    device: str = "cuda"
    out_dir: Path = field(default_factory=lambda: RESULTS)
