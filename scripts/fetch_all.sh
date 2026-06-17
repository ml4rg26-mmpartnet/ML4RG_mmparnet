#!/usr/bin/env bash
# One-shot fetch of everything the demo notebooks need, from PUBLIC sources, into ./data/.
# Designed to run on a fresh clone (e.g. a GPU node). The gated PARNET weights are NOT public;
# the script prints exactly where to drop them.
#
#   bash scripts/fetch_all.sh                 # all RBPs' peaks + hg38 (~3GB)
#   bash scripts/fetch_all.sh --group spliceosome   # just one group's peaks
#   bash scripts/fetch_all.sh --no-hg38       # skip the 3GB genome (e.g. it is already on the node)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ML4RG_DATA:-$HERE/data}"
mkdir -p "$DATA/eclip" "$DATA/refs"
echo "[fetch-all] data dir: $DATA"

GROUP=""; FETCH_HG38="${FETCH_HG38:-1}"
while [ $# -gt 0 ]; do
  case "$1" in
    --group)   GROUP="$2"; shift 2;;
    --no-hg38) FETCH_HG38=0; shift;;
    -h|--help) sed -n '2,11p' "$0"; exit 0;;
    *) echo "[warn] unknown option: $1"; shift;;
  esac
done

# 1. public metadata (shipped in the repo under metadata/) -> data/
for f in eclip_manifest.json cohort.json encode_rbps.json; do
  [ -f "$DATA/$f" ] || cp "$HERE/metadata/$f" "$DATA/$f"
done
echo "[fetch-all] metadata (manifest + cohort + encode_rbps) in place"

# 2. resolve --group to an RBP set (empty = all)
GRP_RBPS=""
if [ -n "$GROUP" ]; then
  GRP_RBPS="$(PYTHONPATH="$HERE/src" ML4RG_DATA="$DATA" python -c "from mmpartnet.io import groups; print(','.join(groups.resolve('$GROUP')))" 2>/dev/null || echo "")"
  echo "[fetch-all] group $GROUP -> ${GRP_RBPS:-<all RBPs>}"
fi

# 3. public ENCODE eCLIP peak BEDs: download by file accession, save under the manifest's basename
#    (so adapters.peaks.resolve_bed finds them). The per-nt bigWig signal is read REMOTELY at run time
#    (adapters.eclip_signal resolves it from the experiment accession), so no signal download here.
PYTHONPATH="$HERE/src" python - "$DATA" "$GRP_RBPS" <<'PY'
import json, sys, pathlib, urllib.request
DATA = pathlib.Path(sys.argv[1])
want = set(filter(None, sys.argv[2].split(","))) if len(sys.argv) > 2 and sys.argv[2] else None
man = json.loads((DATA / "eclip_manifest.json").read_text())
out = DATA / "eclip"; out.mkdir(parents=True, exist_ok=True)
n = skip = 0
for rbp, recs in man.items():
    if want and rbp not in want:
        continue
    for r in (recs if isinstance(recs, list) else [recs]):
        fa = r.get("file"); base = r.get("path") or (f"{fa}.bed.gz" if fa else None)
        if not fa or not base:
            continue
        dst = out / base
        if dst.exists():
            skip += 1; continue
        url = f"https://www.encodeproject.org/files/{fa}/@@download/{fa}.bed.gz"
        try:
            urllib.request.urlretrieve(url, dst); n += 1
        except Exception as e:
            print(f"  [warn] {rbp} {fa}: {e}")
print(f"  downloaded {n} peak BED(s), {skip} already present")
PY

# 4. hg38 (UCSC)
if [ -f "$DATA/hg38.fa" ]; then
  echo "[fetch-all] hg38.fa present"
elif [ "$FETCH_HG38" = "1" ]; then
  echo "[fetch-all] downloading hg38.fa.gz from UCSC (~3GB)..."
  curl -fsSL https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -o "$DATA/hg38.fa.gz"
  gunzip -f "$DATA/hg38.fa.gz"
  PYTHONPATH="$HERE/src" python -c "import pyfaidx; pyfaidx.Faidx('$DATA/hg38.fa')"
else
  echo "[fetch-all] hg38 skipped (--no-hg38); set ML4RG_HG38 to an existing hg38.fa"
fi

# 5. gated PARNET weights (NOT public)
cat <<EOF

[fetch-all] PUBLIC data is ready. The PARNET weights are GATED (ask a supervisor), then place:
    \${ML4RG_REFS:-$DATA/refs}/parnet/models/NewRBPNet_7M_Penalty-0.0_20250107.pt
    \${ML4RG_REFS:-$DATA/refs}/parnet/parnet/assets/ENCODE.idx2symbol-cell.pt
  (or set ML4RG_PARNET_WEIGHTS directly to the .pt).
  Without weights: notebooks 03 (committed JSON) and 04 (offline) still run; 00-02 need the weights.
[fetch-all] done.
EOF
