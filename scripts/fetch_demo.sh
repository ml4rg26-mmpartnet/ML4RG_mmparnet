#!/usr/bin/env bash
# Source-aware fetch for the demo notebook (notebooks/demo/00_mmpartnet_demo.ipynb).
# Pulls only the small slice the chosen DataSource needs, then prints the exact run command.
#
# Usage:
#   bash scripts/fetch_demo.sh                              # encode_bigwig, demo slice (QKI,PTBP1,IGF2BP1)
#   bash scripts/fetch_demo.sh --group AQR --nwin 10        # a different RBP group
#   bash scripts/fetch_demo.sh --source encode_bam_counts   # established counts (prints what's still gated)
#
# encode_bigwig reads the per-nt signal REMOTELY (HTTP range), so the only downloads are the peak BEDs
# (group-scoped) + hg38 + the PARNET refs. Other sources only differ in the OBSERVED target; this script
# tells you what each still needs. Switch source/format in the notebook via cfg.source / cfg.target.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="encode_bigwig"; GROUP="QKI,PTBP1,IGF2BP1"; NWIN="8"
while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    --group)  GROUP="$2";  shift 2;;
    --nwin)   NWIN="$2";   shift 2;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "[warn] unknown option: $1"; shift;;
  esac
done

# point config at on-disk assets if env.local.sh is present (no-op on the GPU node, which exports its own)
[ -f "$HERE/scripts/env.local.sh" ] && { set +u; . "$HERE/scripts/env.local.sh"; set -u; } || true
DATA="${ML4RG_DATA:-$HERE/data}"
echo "[fetch-demo] source=$SOURCE group=$GROUP nwin=$NWIN  data=$DATA"

# always: peak BEDs (group-scoped) + hg38 (opt-in) + ATtRACT note -> reuse the main fetcher
bash "$HERE/scripts/fetch_data.sh" --group "$GROUP"

# source-specific OBSERVED-target requirements
case "$SOURCE" in
  encode_bigwig)
    echo "[fetch-demo] encode_bigwig: observed signal is read remotely (ENCODE bigWig) — no download.";;
  encode_bam_counts)
    echo "[fetch-demo] encode_bam_counts (established 5' crosslink counts): needs the ENCODE GRCh38 eCLIP";
    echo "             alignment BAMs in \$ML4RG_BAMS (large; public on encodeproject.org). LAB-GATED only";
    echo "             by size. Fill sources/encode_bam_counts.observed before running with target=counts.";;
  hfds)
    echo "[fetch-demo] hfds (lab encode.filtered): needs the lab dataset (cfg.extra['path']); LAB-GATED.";;
  local_pt)
    echo "[fetch-demo] local_pt: needs pre-tiled .pt shards in cfg.extra['dir'] (offline/CI handoff).";;
  *) echo "[warn] unknown source '$SOURCE' (known: encode_bigwig|encode_bam_counts|hfds|local_pt)";;
esac

# sanity: confirm hg38 + PARNET weights resolve (the two gated assets the notebook asserts)
PYTHONPATH="$HERE/src" python - <<PY || true
from mmpartnet import config
for label, p in [("HG38", config.HG38), ("PARNET weights", config.PARNET_WEIGHTS),
                 ("eclip_manifest", config.DATA / "eclip_manifest.json")]:
    print(f"  [{'ok' if p.exists() else 'MISS'}] {label}: {p}")
PY

echo "[fetch-demo] ready. Run the demo notebook headlessly with:"
echo "    MMP_SOURCE=$SOURCE MMP_GROUP=$GROUP MMP_NWIN=$NWIN \\"
echo "      python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 \\"
echo "      --output 00_demo_out.ipynb notebooks/demo/00_mmpartnet_demo.ipynb"
echo "  or open it in Jupyter. On a GPU node: bash scripts/run_demos.sh"
