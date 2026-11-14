#!/usr/bin/env bash
# Locked launch for the 变化去模糊 (change-aware deblurring) window.
# The GUI is the primary interface; this script only pins the environment
# (single-threaded numerics, repo-local imports) and opens the browser URL.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

HOST="${SVDB_HOST:-127.0.0.1}"
PORT="${SVDB_PORT:-8765}"

# Fail fast if the numeric stack is missing rather than opening a broken window.
python3 - <<'PY'
import numpy, scipy, PIL
print(f"numpy {numpy.__version__} · scipy {scipy.__version__} · Pillow {PIL.__version__}")
PY

exec python3 -m svdeblur.server --host "$HOST" --port "$PORT"
