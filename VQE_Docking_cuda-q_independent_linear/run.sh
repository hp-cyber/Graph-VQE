#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDAQ_LOCAL="${CUDAQ_LOCAL:-/home/fengy46/cudaq_local}"
VQE_PYTHON="${VQE_PYTHON:-/home/fengy46/conda/envs/cudaq-env/bin/python}"
NCCL_ROOT="${NCCL_ROOT:-$CUDAQ_LOCAL/nccl}"
INDEPENDENT_MPS_INSTALL="$PROJECT_DIR/backend/backend_variants/analytic/install"

if [ ! -s "$CUDAQ_LOCAL/cudaq/lib/libnvqir-tensornet-mps-mqpu-shots.so" ]; then
  echo "Missing custom shots backend in $CUDAQ_LOCAL; set CUDAQ_LOCAL to the reference project's CUDA-Q installation." >&2
  exit 1
fi

if [ ! -s "$INDEPENDENT_MPS_INSTALL/lib/libnvqir-tensornet-mps-mqpu-shots.so" ]; then
  echo "Build the project-local worker-lifecycle fix first: bash $PROJECT_DIR/backend/build.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${VQE_VISIBLE_GPUS:-${CUDA_VISIBLE_DEVICES:-7}}"
export PYTHONPATH="$INDEPENDENT_MPS_INSTALL:$CUDAQ_LOCAL/cudaq:$PROJECT_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$INDEPENDENT_MPS_INSTALL/lib:$CUDAQ_LOCAL/cudaq/lib:$NCCL_ROOT/lib:$(dirname "$VQE_PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export CUDAQ_MPS_NUM_GPUS=1
export CUDAQ_MPS_ABS_CUTOFF=0 CUDAQ_MPS_RELATIVE_CUTOFF=0

cd "$PROJECT_DIR"
if [ "$#" -gt 0 ]; then
  exec "$VQE_PYTHON" "$@"
fi
exec "$VQE_PYTHON" Main.py
