#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_cross_attention_task_ablation.sh pilot [tmux|queue|direct|print]
  scripts/run_cross_attention_task_ablation.sh formal [tmux|queue|direct|print]

Environment overrides:
  PYTHON=/path/to/python
  DEVICE=cuda|cpu
  SEED=0
  OUT_DIR=/path/to/cross_attention_runs
  EPOCHS=15
  STEPS_PER_EPOCH=1000
  PROFILE_STEPS_PER_EPOCH=500
  BATCH_SIZE=32
  ML4RG_REFS=/path/to/parnet_refs
  ML4RG_PARNET_WEIGHTS=/path/to/parnet.7m-0.0.pt

Runs:
  multitask_l10:    profile_loss + 10 * binary_loss, gated target/binary pooling
  profile_only:     profile_loss only, all-positive sampler with half as many steps
  binary_only_l10:  10 * binary_loss, binary head's own position pooling only
USAGE
}

SUITE="${1:-}"
LAUNCH="${2:-tmux}"
if [[ -z "${SUITE}" || "${SUITE}" == "-h" || "${SUITE}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${SUITE}" != "pilot" && "${SUITE}" != "formal" ]]; then
  echo "error: suite must be 'pilot' or 'formal'" >&2
  usage >&2
  exit 2
fi
if [[ "${LAUNCH}" != "tmux" && "${LAUNCH}" != "queue" && "${LAUNCH}" != "direct" && "${LAUNCH}" != "print" ]]; then
  echo "error: launch mode must be 'tmux', 'queue', 'direct', or 'print'" >&2
  usage >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/dgu/venvs/torch39/bin/python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-/home/dgu/cross_attention_runs}"
export ML4RG_REFS="${ML4RG_REFS:-/home/dgu/workspace/parnet_refs}"
export ML4RG_PARNET_WEIGHTS="${ML4RG_PARNET_WEIGHTS:-/home/dgu/storage_ml4rg26-shared/parnet-eclip/models-full-rbp-set/parnet.7m-0.0.pt}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/mmpartnet_pycache}"
export PYTHONPATH="${PYTHONPATH:-src}"

if [[ "${SUITE}" == "pilot" ]]; then
  TRACKS="9,138,195"
  MAX_TRAIN_WINDOWS="512"
  MAX_VALID_WINDOWS="512"
  EPOCHS="${EPOCHS:-2}"
  STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
  PROFILE_STEPS_PER_EPOCH="${PROFILE_STEPS_PER_EPOCH:-50}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  PREFIX="pilot_cross_attention"
else
  TRACKS="all"
  MAX_TRAIN_WINDOWS="0"
  MAX_VALID_WINDOWS="0"
  EPOCHS="${EPOCHS:-15}"
  STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
  PROFILE_STEPS_PER_EPOCH="${PROFILE_STEPS_PER_EPOCH:-500}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  PREFIX="formal_cross_attention"
fi

common_args=(
  --tracks "${TRACKS}"
  --max-train-windows "${MAX_TRAIN_WINDOWS}"
  --max-valid-windows "${MAX_VALID_WINDOWS}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --balanced-train
  --profile-mask-source binding
  --num-blocks 1
  --device "${DEVICE}"
  --seed "${SEED}"
  --out-dir "${OUT_DIR}"
)

run_specs=(
  "multitask_l10|multitask|0.5|1|10|${STEPS_PER_EPOCH}|${PREFIX}_multitask_l10_${EPOCHS}x${STEPS_PER_EPOCH}_seed${SEED}"
  "profile_only|profile-only|1.0|1|0|${PROFILE_STEPS_PER_EPOCH}|${PREFIX}_profile_only_${EPOCHS}x${PROFILE_STEPS_PER_EPOCH}_seed${SEED}"
  "binary_only_l10|binary-only|0.5|0|10|${STEPS_PER_EPOCH}|${PREFIX}_binary_only_l10_${EPOCHS}x${STEPS_PER_EPOCH}_seed${SEED}"
)

quote_cmd() {
  printf '%q ' "$@"
}

build_cmd() {
  local task="$1"
  local pos_fraction="$2"
  local lambda_profile="$3"
  local lambda_binary="$4"
  local steps_per_epoch="$5"
  local run_name="$6"
  CMD=(
    "${PYTHON}" scripts/train_cross_attention_profile.py
    "${common_args[@]}"
    --task "${task}"
    --steps-per-epoch "${steps_per_epoch}"
    --balanced-pos-fraction "${pos_fraction}"
    --lambda-profile "${lambda_profile}"
    --lambda-binary "${lambda_binary}"
    --run-name "${run_name}"
  )
}

launch_one() {
  local label="$1"
  local task="$2"
  local pos_fraction="$3"
  local lambda_profile="$4"
  local lambda_binary="$5"
  local steps_per_epoch="$6"
  local run_name="$7"
  local run_dir="${OUT_DIR}/${run_name}"
  local log_path="${run_dir}/train.log"
  local session="xattn_${SUITE}_${label}_s${SEED}"
  build_cmd "${task}" "${pos_fraction}" "${lambda_profile}" "${lambda_binary}" "${steps_per_epoch}" "${run_name}"

  if [[ "${LAUNCH}" == "print" ]]; then
    echo "# ${session}"
    quote_cmd "cd" "${REPO}"
    echo
    quote_cmd "${CMD[@]}"
    echo
    return
  fi

  mkdir -p "${run_dir}"

  if [[ "${LAUNCH}" == "direct" ]]; then
    echo "running ${label} directly; log=${log_path}"
    (cd "${REPO}" && "${CMD[@]}" 2>&1 | tee "${log_path}")
    return
  fi

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}" >&2
    return
  fi
  local inner
  inner="cd $(printf '%q' "${REPO}") && $(quote_cmd "${CMD[@]}") 2>&1 | tee $(printf '%q' "${log_path}")"
  tmux new-session -d -s "${session}" "${inner}"
  echo "started ${session}; log=${log_path}"
}

launch_queue() {
  local session="xattn_${SUITE}_queue_s${SEED}"
  local queue_cmd="set -euo pipefail; cd $(printf '%q' "${REPO}")"
  for spec in "${run_specs[@]}"; do
    IFS='|' read -r label task pos_fraction lambda_profile lambda_binary steps_per_epoch run_name <<<"${spec}"
    local run_dir="${OUT_DIR}/${run_name}"
    local log_path="${run_dir}/train.log"
    mkdir -p "${run_dir}"
    build_cmd "${task}" "${pos_fraction}" "${lambda_profile}" "${lambda_binary}" "${steps_per_epoch}" "${run_name}"
    queue_cmd+="; echo ===== ${label} =====; $(quote_cmd "${CMD[@]}") 2>&1 | tee $(printf '%q' "${log_path}")"
  done

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}" >&2
    return
  fi
  tmux new-session -d -s "${session}" "${queue_cmd}"
  echo "started sequential queue: ${session}"
  echo "attach with: tmux attach -t ${session}"
}

echo "suite=${SUITE} launch=${LAUNCH} device=${DEVICE} seed=${SEED}"
echo "python=${PYTHON}"
echo "out_dir=${OUT_DIR}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} steps=${STEPS_PER_EPOCH} profile_steps=${PROFILE_STEPS_PER_EPOCH}"
if [[ "${LAUNCH}" == "queue" ]]; then
  launch_queue
  exit 0
fi
for spec in "${run_specs[@]}"; do
  IFS='|' read -r label task pos_fraction lambda_profile lambda_binary steps_per_epoch run_name <<<"${spec}"
  launch_one "${label}" "${task}" "${pos_fraction}" "${lambda_profile}" "${lambda_binary}" "${steps_per_epoch}" "${run_name}"
done
