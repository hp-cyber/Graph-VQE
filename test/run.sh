#!/usr/bin/env bash
# All 660 cases. --dry-run validates the matrix without allocating a GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/environment.sh"
exec "$GRAPH_VQE_PYTHON" "$HERE/tools/run_campaign.py" "$@"
