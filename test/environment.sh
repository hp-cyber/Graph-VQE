#!/usr/bin/env bash
# Source before importing CUDA-Q. Override CUDAQ_LOCAL on another server.
set -euo pipefail
GRAPH_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GRAPH_TEST_ROOT
if [ -z "${GRAPH_VQE_PYTHON:-}" ]; then
  for candidate in "${CONDA_PREFIX:-/nonexistent}/bin/python" "$HOME/conda/envs/cudaq-env/bin/python" "$HOME/miniconda3/envs/cudaq-env/bin/python"; do
    if [ -x "$candidate" ]; then export GRAPH_VQE_PYTHON="$candidate"; break; fi
  done
fi
: "${GRAPH_VQE_PYTHON:?Activate your Python environment or set GRAPH_VQE_PYTHON}"
if [ -z "${CUDAQ_LOCAL:-}" ]; then
  for candidate in "$GRAPH_TEST_ROOT/cudaq_local" "$GRAPH_TEST_ROOT/../cudaq_local" "$GRAPH_TEST_ROOT/../../VQE_Docking_cudaq_tensornet-mps/cudaq_local" "$HOME/cudaq_local"; do
    if [ -f "$candidate/cudaq/analytic_backend_manifest.json" ]; then export CUDAQ_LOCAL="$candidate"; break; fi
  done
fi
: "${CUDAQ_LOCAL:?Build with setup_local_cudaq.sh and source activate_local_cudaq.sh, or set CUDAQ_LOCAL}"
export ANALYTIC_MPS_BACKEND_INSTALL="${GRAPH_VQE_BACKEND_INSTALL:-$CUDAQ_LOCAL/cudaq}"
"$GRAPH_VQE_PYTHON" - "$ANALYTIC_MPS_BACKEND_INSTALL" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);manifest=json.loads((root/'analytic_backend_manifest.json').read_text())
assert manifest['feature']=='analytic-pauli-prefix-v1', 'Accelerated backend required'
name='libnvqir-tensornet-mps-mqpu-shots.so'
assert hashlib.sha256((root/'lib'/name).read_bytes()).hexdigest()==manifest['libraries'][name], 'Backend checksum mismatch'
PY
GRAPH_PYTHON_PREFIX="$(dirname "$(dirname "$GRAPH_VQE_PYTHON")")"
export PATH="$GRAPH_PYTHON_PREFIX/bin:$CUDAQ_LOCAL/cudaq/bin:$PATH"
export PYTHONPATH="$GRAPH_TEST_ROOT/tools:$GRAPH_TEST_ROOT/backend/python_dependencies:$ANALYTIC_MPS_BACKEND_INSTALL:$CUDAQ_LOCAL/cudaq:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ANALYTIC_MPS_BACKEND_INSTALL/lib:$CUDAQ_LOCAL/cudaq/lib:$CUDAQ_LOCAL/nccl/lib:$GRAPH_PYTHON_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
