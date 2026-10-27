#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the INTERACTION dataset root}
PAPER_ASSET_DIR=${PAPER_ASSET_DIR:?Set PAPER_ASSET_DIR to the downloaded GitHub Release assets}

for required_path in   "${DATA_ROOT}/maps"   "${DATA_ROOT}/train"   "${DATA_ROOT}/val"   "${PAPER_ASSET_DIR}/v22_map_geometry_v1.tar.gz"   "${PAPER_ASSET_DIR}/train_style_statistics.pt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

if [[ -f "${PAPER_ASSET_DIR}/SHA256SUMS" ]]; then
  (
    cd "${PAPER_ASSET_DIR}"
    grep -E '  (v22_map_geometry_v1.tar.gz|train_style_statistics.pt)$' SHA256SUMS | sha256sum -c -
  )
else
  echo "Warning: SHA256SUMS is absent; Release assets were not hash-verified." >&2
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/prepare_interaction_data.py"   --root "${DATA_ROOT}"   --split train val

"${PYTHON_BIN}" "${PROJECT_DIR}/compute_csv_motion_cache.py"   --root "${DATA_ROOT}"   --output_dir "${DATA_ROOT}/csv_motion_v1"   --split train val

if [[ -e "${DATA_ROOT}/v22_map_geometry_v1" ]]; then
  echo "Keeping existing map cache: ${DATA_ROOT}/v22_map_geometry_v1"
else
  tar -xzf "${PAPER_ASSET_DIR}/v22_map_geometry_v1.tar.gz"     -C "${DATA_ROOT}"
fi

if [[ -e "${DATA_ROOT}/train_style_statistics.pt" ]]; then
  echo "Keeping existing Style statistics: ${DATA_ROOT}/train_style_statistics.pt"
else
  cp "${PAPER_ASSET_DIR}/train_style_statistics.pt" "${DATA_ROOT}/"
fi

count_samples() {
  local split_name=$1
  find "${DATA_ROOT}/${split_name}_processed" -maxdepth 1 -type f     -name '*.pt' ! -name 'pre_transform.pt' ! -name 'pre_filter.pt' | wc -l
}

train_count=$(count_samples train)
val_count=$(count_samples val)
if (( train_count != 47584 || val_count != 11794 )); then
  echo "Expected 47,584 train and 11,794 validation scenarios; found ${train_count}/${val_count}." >&2
  exit 1
fi

map_count=$(find "${DATA_ROOT}/v22_map_geometry_v1" -maxdepth 1 -type f -name '*.pt' | wc -l)
if (( map_count != 18 )); then
  echo "Expected 18 V22 map geometry files; found ${map_count}." >&2
  exit 1
fi

echo "Paper data are ready: train=${train_count}, val=${val_count}, map_files=${map_count}"
