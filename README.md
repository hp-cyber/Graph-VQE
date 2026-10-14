# Graph-VQE

The matched campaign in `test/` runs Hamiltonian-aware independent VQE for 55
proteins, both linear and reverse-linear entanglement, COBYLA/L-BFGS-B/SPSA, and
both analytic (`shots=-1`) and 1000-shot objectives: 660 cases.

Run `./test/run.sh`; use `./test/run.sh --dry-run` to preview the pool. The default
is five workers per GPU on physical GPUs 2,3,4,5,6, with immediate refill when a
worker finishes. `--gpus`, `--jobs-per-gpu`, `--proteins`, `--optimizers`, `--shots`
and `--entanglements` select resources or cases. GPU validation tools default to
GPU 2; GPUs 0 and 1 are not used by the default campaign. For isolated timing
measurements, explicitly use `--jobs-per-gpu 1` and otherwise idle GPUs.

To keep growing logs on a larger disk, export `CAMPAIGN_RECORDS_ROOT` to a
writable directory, or copy `test/campaign_storage.env.example` to the ignored
`test/campaign_storage.env` and edit it. The launcher sources that optional file.
Checkpoints stay at their original case paths; JSONL/CSV logs and `run.log` are
linked to the data disk after existing contents are copied and SHA256-verified.
Each case's `record_storage.json` remembers its destination on restart. There is
no added disk quota, log-size cap or free-space reserve. Both filesystems need
available space. Use `test/tools/clean_unfinished.py --apply` for cleanup so that
the selected case's external logs are removed too.

An interrupted final JSONL write is archived before another worker appends;
earlier complete records and checkpoints are retained. Memory-summary writes are
atomic, and a shutdown-reporting failure does not replace the original error.

Each partition is optimized once, its best rotation parameters initialize the
global nearest-neighbor circuit, and the full Hamiltonian is optimized once.
Early stopping uses consecutive energy change below 1, without patience or a
window. The existing gradient thresholds and explicit iteration caps remain.
Partitioning, Hamiltonians, circuits, and optimizer mathematics are unchanged by
the work-recording layer.

Every case under `test/runs/` records:

- `work_events.jsonl`: actual energies, parameters and parameter counts, durations,
  attempt IDs, parent events, local partition IDs and qubits, warm-start/global
  stages, objective/probe roles, optimizer iterations, reverse gradients and readout.
- `work_summary.json`: exact completed operation counts and inclusive times by
  partition/stage/role, checked against the separate backend and optimizer logs.
- `energy_evaluations.csv` and `optimizer_iterations.csv`: scalar energies and
  completed optimizer iterations with their parameters and elapsed times. An
  unfinished iteration tail is labeled separately and is not a completed iteration.
- `global_energy_timeline.jsonl`: actual full-Hamiltonian energies in chronological
  order across attempts, with restart downtime excluded from active elapsed time.
  Local partition energies are not summed into a global curve: the Hamiltonian
  can contain crossing terms.
- `optimizer_states/` and `solves/`: atomically saved latest parameters and retained
  best states. Restart restores unfinished local/global solves and reuses completed
  local solves. SciPy rebuilds its simplex/Hessian; this is a parameter warm restart,
  not a bit-for-bit continuation of SciPy's internal state. SPSA also restores gains
  and RNG state.
- `memory_usage.json`: sampled RAM/GPU-memory peaks and confirmation that the
  worker's process group was cleaned after exit.

On resume, a `local_solve` stage can return a saved result without invoking an
optimizer; its record has `reused_completed_solve=true`. Count actual optimizer
invocations using `optimizer_solve`, not the number of stage entries.

Each worker handles one case and exits. The scheduler reaps it and kills any
surviving children in that worker's process group before refilling the slot, so
process-owned RAM and CUDA contexts are released. Memory budgets can stop an
oversized worker; there are no project-imposed elapsed-time limits, including
GPU monitoring calls. Ctrl+C stops the launcher and its workers. Completed cases
are skipped; cases with a work ledger require valid, complete work records before
being considered finished. A truncated final ledger write is archived and repaired
on restart without dropping previous complete records.

Nested timings overlap: do not add iteration, energy, backend and gradient
durations together. With five jobs per GPU, measured wall times include contention.
Counts can be combined with separately measured per-operation costs to estimate
another server's runtime, but those estimates are not measured execution times.

Analytic objectives retain shared-prefix MPS acceleration and L-BFGS retains reverse
gradient acceleration. With 1000 shots, objective/readout sampling adds statistical
noise; L-BFGS uses the existing analytic surrogate gradient. No extra hardware
error model is added. The four-layer nearest-neighbor circuits retain their
circuit-derived exact bond bound (up to 16), with zero SVD cutoffs.

After `source test/environment.sh`, CPU checks run with
`python -m unittest discover -s test/tests -v`. Audit a finished case with
`python test/tools/validate_work_records.py CASE`. Restart tests support
`python test/tools/validate_resume.py --gpu 2 --shots -1 --phase local --output OUTPUT`
(also `--shots 1000` and `--phase global`). Choose a fresh output directory for each
validation batch. Validation artifacts are separate from campaign results.
