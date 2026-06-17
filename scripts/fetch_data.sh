#!/usr/bin/env bash
# Idempotent fetch of the EXTERNAL assets the pipeline needs into ./data/ (gitignored).
# Nothing here is committed — large/licensed assets stay external.
#
# Usage:
#   bash scripts/fetch_data.sh --demo        # NO downloads: verify the offline synthetic fixture
#   bash scripts/fetch_data.sh               # pull the small/automatable assets; print gated steps
#   FETCH_HG38=1 bash scripts/fetch_data.sh  # ALSO download hg38.fa (~3GB) from UCSC
#
# Populates:
#   data/hg38.fa (+ .fai)                  human genome (pyfaidx)            [opt-in: FETCH_HG38=1]
#   data/ATtRACT_db.txt , data/pwm.txt     ATtRACT RBP motif DB             [form-gated: manual]
#   data/eclip/*.bed.gz                    ENCODE eCLIP peak BEDs           [auto if manifest has urls]
#   data/embeddings_all.npz , pe_string.npz  -> scripts/build_embeddings.py [needs .[esm] extra]
#   data/refs/parnet*                      lab PARNET pkg + weights         [gated access: manual]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ML4RG_DATA:-$HERE/data}"
mkdir -p "$DATA/refs" "$DATA/eclip"
echo "data dir: $DATA"

# --- parsed options: --demo, --group <name> (which RBP group's data to fetch) ---
DEMO=""; GROUP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --demo)  DEMO=1; shift;;
    --group) GROUP="$2"; shift 2;;
    --list)  PYTHONPATH="$HERE/src" ML4RG_DATA="$DATA" python -c "from mmpartnet.io import groups; print(groups.list_str())"; exit 0;;
    *) echo "[warn] unknown option: $1"; shift;;
  esac
done
GRP_RBPS=""
if [ -n "$GROUP" ]; then
  GRP_RBPS="$(PYTHONPATH="$HERE/src" ML4RG_DATA="$DATA" python -c "from mmpartnet.io import groups; print(','.join(groups.resolve('$GROUP')))" 2>/dev/null || echo "")"
  echo "[group] $GROUP -> ${GRP_RBPS:-<all RBPs>}"
fi

# --- import check: prove the package imports on a fresh clone (no external assets) ---
if [ "${DEMO:-}" = "1" ]; then
  echo "[demo] verifying the package imports (no external assets) ..."
  PYTHONPATH="$HERE/src" python -c "import mmpartnet.data, mmpartnet.protein, mmpartnet.splits, mmpartnet.m2; print('mmpartnet imports OK:', mmpartnet.data.list_sources())"
  echo "[demo] then: bash scripts/fetch_all.sh  &&  bash scripts/run_demos.sh"
  exit 0
fi

# --- hg38 (opt-in; ~3GB) ---
if [ -f "$DATA/hg38.fa" ]; then
  echo "[skip] hg38.fa present"
elif [ "${FETCH_HG38:-}" = "1" ]; then
  echo "[fetch] hg38.fa (~3GB) from UCSC ..."
  curl -L https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -o "$DATA/hg38.fa.gz"
  gunzip -f "$DATA/hg38.fa.gz"
  python -c "import pyfaidx; pyfaidx.Faidx('$DATA/hg38.fa')"
else
  echo "[note] hg38.fa absent — set FETCH_HG38=1 to download (~3GB), or use --demo"
fi

# --- ATtRACT motif DB (form-gated; manual download) ---
if [ -f "$DATA/ATtRACT_db.txt" ] && [ -f "$DATA/pwm.txt" ]; then
  echo "[skip] ATtRACT present"
else
  echo "[note] ATtRACT: download ATtRACT_db.txt + pwm.txt from https://attract.cnic.es/ into $DATA"
fi

# --- public ENCODE eCLIP peak BEDs (manifest-driven; auto-fetch any entry with a 'url') ---
MAN="$DATA/eclip_manifest.json"
if [ -f "$MAN" ]; then
  echo "[fetch] ENCODE eCLIP BEDs per $MAN (entries with a 'url' field${GRP_RBPS:+; group=$GROUP}) ..."
  python - "$MAN" "$DATA/eclip" "$GRP_RBPS" <<'PY'
import json, sys, pathlib, urllib.request
man = json.loads(open(sys.argv[1]).read()); out = pathlib.Path(sys.argv[2]); n = 0
want = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None
for rbp, recs in man.items():
    if want and rbp not in want:
        continue
    for rec in (recs if isinstance(recs, list) else [recs]):
        url = rec.get("url") if isinstance(rec, dict) else None
        if not url:
            continue
        dst = out / pathlib.Path(url).name
        if dst.exists():
            continue
        try:
            urllib.request.urlretrieve(url, dst); n += 1
        except Exception as e:
            print(f"  [warn] {rbp}: {e}")
print(f"  fetched {n} new BED file(s)")
PY
else
  echo "[note] no eclip_manifest.json — build it (RBP -> ENCODE accession/url) from https://www.encodeproject.org/"
fi

# --- protein embeddings (built, not downloaded) ---
echo "[note] protein reps: python scripts/build_embeddings.py   (needs the .[esm] extra)"
# --- lab PARNET fork + scaffold (gated access; CONFIRM with a supervisor) ---
echo "[note] clone mhorlacher/parnet + parnet--demo--train-models into $DATA/refs/ ; set ML4RG_PARNET_WEIGHTS"
echo "done."
