#!/usr/bin/env bash
# Shared environment for every release entry point.  Sourced, never executed.
# Guarantees one thread per numeric backend and a repo-rooted PYTHONPATH.
set -euo pipefail

SVDB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SVDB_ROOT
cd "$SVDB_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONPATH="$SVDB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
