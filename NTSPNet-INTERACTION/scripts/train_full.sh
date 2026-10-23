#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODULES [SEED]" >&2
  echo "MODULES: all, style, reachability, priority, style_reachability, style_priority, reachability_priority" >&2
  exit 2
fi

MODULES=$1
SEED=${2:-2048}
case "${MODULES}" in
  all|style|reachability|priority|style_reachability|style_priority|reachability_priority) ;;
  *) echo "Unsupported module selection: ${MODULES}" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the INTERACTION dataset root}
BASE_WEIGHTS=${BASE_WEIGHTS:?Set BASE_WEIGHTS to hpnet_local_baseline_v22.ckpt}
PROVIDER_WEIGHTS=${PROVIDER_WEIGHTS:?Set PROVIDER_WEIGHTS to semantic_provider_v14.ckpt}
CSV_MOTION_CACHE_DIR=${CSV_MOTION_CACHE_DIR:-${DATA_ROOT}/csv_motion_v1}
V22_MAP_GEOMETRY_CACHE_DIR=${V22_MAP_GEOMETRY_CACHE_DIR:-${DATA_ROOT}/v22_map_geometry_v1}
STYLE_STATISTICS_PATH=${STYLE_STATISTICS_PATH:-${DATA_ROOT}/train_style_statistics.pt}
GPU_ID=${GPU_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
RUN_LOG_ROOT=${RUN_LOG_ROOT:-${PROJECT_DIR}/full_run_logs}
RUN_LOG=${RUN_LOG:-${RUN_LOG_ROOT}/ntspnet_${MODULES}_seed${SEED}.log}

for required_path in \
  "${BASE_WEIGHTS}" \
  "${PROVIDER_WEIGHTS}" \
  "${STYLE_STATISTICS_PATH}" \
  "${V22_MAP_GEOMETRY_CACHE_DIR}" \
  "${DATA_ROOT}/train_processed" \
  "${DATA_ROOT}/val_processed" \
  "${CSV_MOTION_CACHE_DIR}/train" \
  "${CSV_MOTION_CACHE_DIR}/val"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

count_samples() {
  local split_name=$1
  find "${DATA_ROOT}/${split_name}_processed" -maxdepth 1 -type f \
    -name '*.pt' ! -name 'pre_transform.pt' ! -name 'pre_filter.pt' | wc -l
}

train_count=$(count_samples train)
val_count=$(count_samples val)
if (( train_count != 47584 || val_count != 11794 )); then
  echo "Expected 47,584 train and 11,794 validation scenes; found ${train_count}/${val_count}." >&2
  exit 1
fi

mkdir -p "${RUN_LOG_ROOT}"
if [[ -e "${RUN_LOG}" ]]; then
  echo "Refusing to overwrite existing log: ${RUN_LOG}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
echo "NTSPNet full-data training: modules=${MODULES}, seed=${SEED}, train=${train_count}, val=${val_count}"
echo "Physical batch=2, gradient accumulation=8, effective batch=16, epochs=1"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" train.py \
  --seed "${SEED}" \
  --root "${DATA_ROOT}" \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --devices 1 \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --max_epochs 1 \
  --accumulate_grad_batches 8 \
  --base_weights_path "${BASE_WEIGHTS}" \
  --residual_weights_path "${PROVIDER_WEIGHTS}" \
  --style_statistics_path "${STYLE_STATISTICS_PATH}" \
  --csv_motion_cache_dir "${CSV_MOTION_CACHE_DIR}" \
  --v22_map_geometry_cache_dir "${V22_MAP_GEOMETRY_CACHE_DIR}" \
  --lr 2e-4 \
  --weight_decay 0 \
  --warmup_epochs 1 \
  --T_max 1 \
  --flip_p 0.5 \
  --agent_occlusion_ratio 0.0 \
  --lane_occlusion_ratio 0.0 \
  --trunk_variant original_residual \
  --baseline_residual_max_scale 0.1 \
  --baseline_residual_training_scope fusion_only \
  --baseline_residual_modules "${MODULES}" \
  --motion_source csv \
  --no_preserve_base_edges \
  --reachability_residual_scale 0 \
  --style_film_mode bounded_identity \
  --style_film_bound 0.1 \
  --conflict_update_scope legacy \
  --priority_injection_mode legacy \
  --proposal_conflict_mode_scale 0 \
  --proposal_conflict_mode_adapter_scale 0 \
  --proposal_conflict_residual_gate_scale 0 \
  --scene_mode_coordination_scale 0 \
  --joint_mode_composition_scale 0 \
  --proposal_conflict_training_scope all \
  --lambda_hist_aux 0 \
  --lambda_priority 0 \
  --lambda_style 0 \
  --lambda_endpoint 0 \
  --lambda_late_horizon 0 \
  --lambda_teacher_distillation 0 \
  2>&1 | tee "${RUN_LOG}"
