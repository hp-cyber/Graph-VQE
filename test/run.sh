#!/usr/bin/env bash
# All 660 cases. No arguments needed: GPUs are GRAPH_VQE_GPUS if set (this server: 2,3,4,5,6 in
# campaign_storage.env), else every GPU nvidia-smi lists. Jobs start one at a time while the host stays
# responsive and the GPU's CPU socket is not full; at 90% RAM (or of the container limit) every job stops
# and later resumes from its checkpoint (tools/admission.py). Workers run at low CPU/disk priority on the
# GPU's socket. Rerunning skips completed cases and resumes the others.
# --dry-run validates the matrix without allocating a GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/environment.sh"
if [[ -f "$HERE/campaign_storage.env" ]]; then source "$HERE/campaign_storage.env"; fi
# Scheduler state; launchers sharing it share one start rate and RAM/responsiveness policy.
export ANALYTIC_MPS_ADMISSION_DIR="${ANALYTIC_MPS_ADMISSION_DIR:-$HERE/admission}"
# A directory copied from another server can carry links/paths that do not exist here.
if [[ -L "$HERE/runs" && ! -e "$HERE/runs" ]]; then
  echo "ERROR: $HERE/runs links to missing $(readlink "$HERE/runs"); create that directory or remove the link." >&2; exit 1
fi
for dir in "$ANALYTIC_MPS_ADMISSION_DIR" ${CAMPAIGN_RECORDS_ROOT:+"$CAMPAIGN_RECORDS_ROOT"}; do
  if ! mkdir -p "$dir" 2>/dev/null || [[ ! -w "$dir" ]]; then
    echo "ERROR: cannot write $dir; edit or remove $HERE/campaign_storage.env." >&2; exit 1
  fi
done
exec "$GRAPH_VQE_PYTHON" "$HERE/tools/run_campaign.py" "$@"
