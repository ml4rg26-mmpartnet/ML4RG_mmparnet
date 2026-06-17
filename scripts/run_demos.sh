#!/usr/bin/env bash
# Execute the demo notebooks headlessly -> notebooks/demo/executed/*_executed.ipynb.
# Self-contained: no external dispatcher needed. Run after scripts/fetch_all.sh (+ PARNET weights for 00-02).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$HERE/scripts/env.local.sh" ] && { set +u; . "$HERE/scripts/env.local.sh"; set -u; } || true
export ML4RG_DATA="${ML4RG_DATA:-$HERE/data}"
export ML4RG_HG38="${ML4RG_HG38:-$ML4RG_DATA/hg38.fa}"
export ML4RG_REFS="${ML4RG_REFS:-$ML4RG_DATA/refs}"
export PYTHONPATH="$HERE/src"
# demo slice knobs (override as needed)
export MMP_GROUP="${MMP_GROUP:-spliceosome}" MMP_NWIN="${MMP_NWIN:-16}" MMP_EPOCHS="${MMP_EPOCHS:-8}"
export MMP_FT_GROUP="${MMP_FT_GROUP:-SF3B4,U2AF2,PRPF8,AQR,RBM22}" MMP_FT_NWIN="${MMP_FT_NWIN:-12}"
OUT="$HERE/notebooks/demo/executed"; mkdir -p "$OUT"
python -m ipykernel install --user --name python3 --display-name "Python 3" >/dev/null 2>&1 || true

ok=0; fail=0
for nb in "$HERE"/notebooks/demo/0*.ipynb; do
  base=$(basename "$nb" .ipynb)
  echo "=== executing $base ==="
  if python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=5400 \
       --ExecutePreprocessor.kernel_name=python3 --output-dir "$OUT" --output "${base}_executed.ipynb" "$nb"; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1)); echo "[warn] $base FAILED (continuing; 00-02 need PARNET weights)"
  fi
done
echo "executed ok=$ok fail=$fail -> $OUT"
