#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the INTERACTION dataset root}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to a released .ckpt file}
CSV_MOTION_CACHE_DIR=${CSV_MOTION_CACHE_DIR:-${DATA_ROOT}/csv_motion_v1}
V22_MAP_GEOMETRY_CACHE_DIR=${V22_MAP_GEOMETRY_CACHE_DIR:-${DATA_ROOT}/v22_map_geometry_v1}
GPU_ID=${GPU_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}

cd "${PROJECT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" val.py \
  --root "${DATA_ROOT}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --devices 1 \
  --ckpt_path "${CHECKPOINT}" \
  --csv_motion_cache_dir "${CSV_MOTION_CACHE_DIR}" \
  --v22_map_geometry_cache_dir "${V22_MAP_GEOMETRY_CACHE_DIR}"
