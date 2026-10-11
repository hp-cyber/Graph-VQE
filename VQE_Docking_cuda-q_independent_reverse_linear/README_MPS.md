Run the independent Hamiltonian-aware VQE workflow:

```bash
./run.sh
# Physical GPUs (default: 4,5):
VQE_VISIBLE_GPUS=0,1,2,3 ./run.sh
# Correctness checks and a small real-protein integration run:
./run.sh validate_independent_vqe.py
```

The launcher loads `/home/fengy46/cudaq_local`, the custom CUDA-Q installation
used by the sieve-VQE reference project. Set `CUDAQ_LOCAL` and `VQE_PYTHON` to
override the installation and interpreter. `run.sh` overlays the project-local
backend in `backend/backend_variants/analytic/install`. It fixes a native
worker-lifecycle bug: resizing a circuit could cause new GPU workers to execute
a stale job containing freed pointers. The fix also rebuilds worker pools when
GPU groups change. The shared CUDA-Q installation is unchanged. Source and a
small reviewable diff are in `backend/source/` and
`backend/worker_lifecycle.patch`. Rebuild with `bash backend/build.sh` (requires
the local CUDA-Q source/build tree and CUDA toolchain). The target is
`tensornet-mps-mqpu-shots`; no MPI launch is required.

`Main.py` runs four strategies using `IndependentSubspaceVQE`: linear,
reverse_linear, circular, and sca, each with four local circuit repetitions.
Energy observations and final bitstring readout use 1000 shots. This introduces
sampling noise; no additional gate or readout-error channel is configured.

The algorithm is:

1. Build a weighted interaction graph from the Hamiltonian. Every pair in a
   Pauli term's support receives the absolute value of that term's coefficient.
   Partition connected components with Louvain communities. Communities larger
   than `max_block_qubits=12` are split by weighted balanced cuts. This setting
   controls local workload; it does not limit the full register size. Set it to
   `None` in `Main.py` to retain entire communities.
2. Give each block its own smaller circuit and within-block Hamiltonian,
   reindexing qubits correctly. Assign the identity constant exactly once.
   Optimize these independent objectives on disjoint GPU groups. Single-qubit
   blocks have one idle spectator because this backend requires two qubits;
   spectators are absent from the combined logical circuit. Constant-only local
   objectives require one energy observation and no optimization.
3. Prepare the tensor product of the optimized local circuits on their original
   qubits. Preserve each local circuit's pattern and gate order, including
   noncontiguous blocks. Add parameterized ZY and YZ Pauli rotations on every
   cross-block graph edge, initialized at zero so they act as identity. These
   gates can change populations as well as introduce entanglement for diagonal
   protein Hamiltonians.
4. Evaluate the combined state against the **complete original Hamiltonian**,
   including every crossing Pauli term. Refine all local rotation parameters
   and cross-block rotation parameters jointly. Locals are trained once, and
   global passes continue from the best full-Hamiltonian candidate without
   resetting the cross-block parameters. The initial product state remains a
   candidate for the returned best state.

Local optimization is an approximate warm start whenever crossing terms exist.
It is an exactly separable objective only when no terms cross the partition.
Neither local optimization nor global VQE guarantees the ground state. The
global ansatz is the product circuit plus trainable cross-block gates;
substituting the old full-register fixed-CNOT circuit would generally destroy
the optimized product state at initialization.

Early stopping uses an **absolute energy improvement threshold of 1.0**.
After each accepted local or global optimizer step, stop immediately if
`previous_accepted_energy - current_energy < 1`, including a worsening energy.
The first accepted step is compared with the initial objective value. Rejected
line-search trials do not control stopping. There are no patience counters,
rolling averages, or smoothing windows. A global pass also stops the outer
loop immediately when its step stopping rule fires or its best improvement is
below 1. Numerical solver convergence, failure, and iteration limits may finish
optimization earlier; L-BFGS-B numerical tolerances are separate from this
absolute energy rule. Parameter-shift gradients also use 1000-shot observations.

Each strategy writes these files under
`Result/process_data/<strategy>/<protein>/`:

- `stage_summary.json`: partition, local circuit widths, within/crossing term
  counts, crossing coefficient weight, local parameters and energies, stopping
  reasons, product-state full energy, GPU workers, bond caps, and timing totals.
- `System_Energy/energy_evaluation_timings.csv`: every observation, including
  local objectives and gradient shifts, with phase (`local`, `product`,
  `global`), block qubits, QPU, shots, energy, start offset, and elapsed seconds.
- `System_Energy/iteration_timings.csv`: each accepted optimizer interval and
  its evaluation count and duration. `early_stop` marks the triggering step;
  `optimizer_exit` covers remaining work without another accepted callback.
- `System_Energy/epoch_timings.csv`: full global-pass duration and evaluation
  totals. The energy trace and best-results list contain **only full-Hamiltonian
  energies**; local energies are recorded separately in the JSON.

Durations use a monotonic clock and include synchronized CUDA-Q retrieval,
runtime overhead, and GPU queue time. They are not CUDA kernel profiler times.
CSV writes are locked and flushed at callbacks. Summed evaluation durations
can exceed elapsed wall time when locals run concurrently. GPU group and bond
settings change only between phases after all workers have joined. Extra GPUs
shard the fixed shot budget; global optimization and readout use all selected
GPUs in one group.

There is no configured total-qubit ceiling. Both MPS singular-value cutoffs are
zero. Local circuits use a shared full-rank bound for the largest local width;
the combined global circuit uses the full arbitrary-state bound
`2**(num_qubits//2)`. **The old linear bond-16 claim for 97 qubits does not apply
to this composed global circuit**, even when local blocks use linear ansätze.
Noncontiguous blocks and cross-block gates can require much higher ranks.
The custom backend represents ranks with signed 32-bit integers; larger bounds
are capped at its representable maximum, 2147483647. Exact MPS evolution up to
floating-point roundoff requires actual ranks to fit that cap and GPU memory.
No deliberate singular-value truncation is configured, but arbitrary large
states cannot be promised lossless simulation beyond those physical limits.
Independent locals reduce local simulation cost; global optimization can still
dominate and total speedup is workload dependent.

`validate_independent_vqe.py` checks partition reconstruction, singleton/constant
objectives, all four circuit patterns, preservation of product states using
exact local projectors, energies and cross/local gradients against Qiskit,
accepted-step stopping, parallel execution, timing consistency, and all four
real-protein pipelines including sampled readout and XYZ export. It writes
`validation_independent_vqe.json`. Earlier `validate_subspace_*` scripts and
`validation_hamiltonian_subspace.json` concern the inherited full-register block
optimizer, not this independent implementation.
