#!/bin/bash
# Launcher for VQE_Docking on the locally-built tensornet-mps-mgpu backend.
#
#   ./run.sh            # runs Main.py
#   ./run.sh foo.py     # runs foo.py instead
#
# Sets up the from-source CUDA-Q (cudaq_local) + NCCL + GPUs 4,5,6.
set -e

# --- activate conda env (provides python, cuQuantum, cmake) -------------------
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate cudaq-env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- locate the from-source CUDA-Q install (cudaq_local) ----------------------
# Prefer a project-local build (created by setup_local_cudaq.sh), else the known
# working build under ~/huangj84/cudaq_local.
CUDAQ_LOCAL="${CUDAQ_LOCAL:-}"
if [ -z "$CUDAQ_LOCAL" ]; then
  for d in "$SCRIPT_DIR/../cudaq_local" "$HOME/huangj84/cudaq_local"; do
    [ -d "$d/cudaq/cudaq" ] && { CUDAQ_LOCAL="$d"; break; }
  done
fi
if [ -z "$CUDAQ_LOCAL" ] || [ ! -d "$CUDAQ_LOCAL/cudaq/cudaq" ]; then
  echo "ERROR: could not find cudaq_local. Run setup_local_cudaq.sh first, or set CUDAQ_LOCAL." >&2
  exit 1
fi
NCCL_ROOT="${NCCL_ROOT:-$CUDAQ_LOCAL/nccl}"
echo ">> using CUDA-Q build: $CUDAQ_LOCAL"

# --- environment for the mgpu backend ----------------------------------------
# The system libnvidia-ml.so (580.159.03) is newer than the kernel module
# (580.119.02), which causes nvmlInit_v2 to fail inside NCCL. Prepend the
# user-local NVIDIA install that exactly matches the loaded kernel module.
NVIDIA_LOCAL_LIB="$HOME/huangj84/.local/nvidia-580.119.02/usr/lib/x86_64-linux-gnu"
export PYTHONPATH="$CUDAQ_LOCAL/cudaq:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$NVIDIA_LOCAL_LIB:$CUDAQ_LOCAL/cudaq/lib:$NCCL_ROOT/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="6"
# Single-GPU MPS mode: each circuit uses exactly 1 GPU; parallel SubspaceVQE
# blocks land on GPUs 4, 5, 6 in round-robin order without NCCL overhead.
export CUDAQ_MPS_NUM_GPUS=1

cd "$SCRIPT_DIR"
python "${1:-Main.py}" "${@:2}"
