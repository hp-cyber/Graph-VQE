# Independent Hamiltonian-aware VQE

Run `./run.sh`. This directory selects `reverse_linear`, with four repetitions,
COBYLA and 1,000 shots for energy evaluation and final readout. The matched
campaign with both shot modes and all three optimizers is in `../test`.

The Hamiltonian's coefficient-weighted interaction graph defines partitions
of at most 12 qubits. Each partition is optimized once. Its best rotation
parameters are mapped into the full-register circuit by layer, physical qubit
and rotation gate. Then one global optimization uses the complete Hamiltonian,
including all cross-partition terms.

Both stages use nearest-neighbor `reverse_linear` SU2 circuits. The global circuit
has no extra graph-edge gates and uses `n * (reps + 1) * 2` parameters: 970 at
97 qubits. Parameter transfer supplies an initial guess; it does not reproduce
the exact tensor product of the locally optimized states. The graph still
controls partitioning, but no longer controls global entangling gates.

Each local/global solve stops when an accepted step improves energy by less
than 1, including a worsening step. There is no patience or smoothing window.
Solver convergence and iteration limits can also end a solve. There is no
alternating local/global outer loop. The matched test campaign uses its common
consecutive-energy stopping rule as documented in `../test/USAGE.txt`.

The MPS cap for each stage is `min(2**reps, 2**floor(width/2))`: 16 at 97 qubits,
8 at 7 qubits. Absolute/relative cutoffs are zero. This cap preserves the states
of this nearest-neighbor ansatz up to floating-point error; it does not preserve
the variational family of the former graph-edge global circuit.

The local backend includes the worker-lifecycle fix and stable direct SVD with
an insufficient-bond guard. Rebuild using `bash backend/build.sh`. `run.sh`
loads this local plugin ahead of the shared CUDA-Q installation. Set
`CUDAQ_LOCAL`, `VQE_PYTHON` and `VQE_VISIBLE_GPUS` for another installation.
The target is `tensornet-mps-mqpu-shots`.

Outputs under `Result/process_data/<strategy>/<protein>/` include parameters,
partition metadata, local/global energies and stopping reasons, bond caps,
and synchronized evaluation/iteration timing CSVs. The initial global energy
is named `warm_start_energy`; timing phases are `local`, `warm_start`, `global`.
Old graph-gate results/checkpoints are incompatible and require a fresh run.

Validate with `./run.sh validate_independent_vqe.py`. Checks cover Hamiltonian
reconstruction, noncontiguous partitions, global-chain energies and gradients
against a dense reference, local parameter mapping, stopping, timing, sampled
readout, and a bounded real-protein pipeline.
