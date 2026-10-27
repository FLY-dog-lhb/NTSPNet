#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_DIR}"

while IFS= read -r -d '' script_path; do
  bash -n "${script_path}"
done < <(find scripts -type f -name '*.sh' -print0)

"${PYTHON_BIN}" -m compileall -q   datamodules datasets layers losses metrics model modules transforms utils   ./*.py
"${PYTHON_BIN}" -m pytest tests -q

if [[ -n "${CHECKPOINT:-}" ]]; then
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - "${CHECKPOINT}" <<'PY'
import contextlib
import io
from pathlib import Path
import sys

from model import PNet

checkpoint = Path(sys.argv[1]).expanduser().resolve()
captured_stdout = io.StringIO()
captured_stderr = io.StringIO()
with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
    model = PNet.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        map_location='cpu',
        strict=True,
    )
print(f'strict checkpoint load: OK ({type(model).__name__})')
PY
fi

echo "Repository reproduction checks: OK"
