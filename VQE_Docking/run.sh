#!/bin/bash
# Launcher for VQE_Docking_cudaq_copy using editable cudaq source.
#
# Editable cudaq Python source: ~/cudaq_source/cudaq/
# Compiled .so runtime:         ~/conda/envs/cudaq-env/ (symlinked)
#
# Edit Python files in ~/cudaq_source/cudaq/ and changes take effect immediately.

set -e

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate cudaq-env

# Point to compiled cudaq shared libraries
export LD_LIBRARY_PATH="$HOME/conda/envs/cudaq-env/lib/python3.11/site-packages/lib:${LD_LIBRARY_PATH:-}"

# Use editable cudaq source instead of installed package
export PYTHONPATH="$HOME/cudaq_source:${PYTHONPATH:-}"

cd "$(dirname "$0")"

# Run Main.py (or pass custom script as argument)
python "${1:-Main.py}" "${@:2}"
