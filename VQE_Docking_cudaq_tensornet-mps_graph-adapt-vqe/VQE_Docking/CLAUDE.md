# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VQE_Docking is a quantum computing pipeline for protein structure prediction and ligand docking. It uses the Variational Quantum Eigensolver (VQE) to fold short peptide sequences on a lattice model, then post-processes the predicted structures for molecular docking with AutoDock Vina.

The project supports two quantum backends:
- **CUDA-Q (primary)**: GPU-accelerated simulation via NVIDIA's CUDA-Q SDK, with multi-GPU support (mgpu for memory pooling, mqpu for parallel blocks)
- **IBM Quantum (legacy)**: Qiskit Runtime on real IBM hardware (not actively maintained)

## Running

```bash
# Run the main pipeline (requires CUDA-Q and GPUs)
python Main.py

# MPI launch for multi-GPU (mgpu backend)
mpirun -np <num_gpus> python Main.py
```

Key environment: `CUDA_VISIBLE_DEVICES` is set in `Main.py` (default: "4, 5"). Edit this for your GPU setup.

## Architecture

### Pipeline Flow
1. **Hamiltonian Construction** (`Protein_Folding/`) — Encodes a peptide sequence into a qubit Hamiltonian using Miyazawa-Jernigan interaction energies on a lattice model. Outputs a Qiskit `SparsePauliOp`.
2. **Hamiltonian Conversion** (`Qiskit_VQE/cudaq.py:qiskit_sparsepauliop_to_cudaq_spin`) — Converts Qiskit operator to CUDA-Q `SpinOperator` with qubit order reversal.
3. **VQE Optimization** (`Qiskit_VQE/`) — Finds the ground state using parameterized quantum circuits (EfficientSU2 ansatz).
4. **Measurement** (`Qiskit_VQE/measure.py:CudaqStateCalculator`) — Extracts probability distributions from optimized circuits (statevector or shot-based).
5. **Structure Interpretation** (`Protein_Folding/protein_folding_result.py`) — Decodes the most probable bitstring into 3D coordinates and saves `.xyz` files.
6. **Post-Processing** (`Post_Processing/`, `Result/`) — Reconstructs full atomic models (via Modeller), prepares docking files (PDBQT), and runs AutoDock Vina scoring.

### VQE Strategies (configured in Main.py)
- **Baseline** (`CUDAQVQE`): Single-stage VQE with one entanglement pattern (full, linear, circular, pairwise, sca, reverse_linear)
- **EfficientNet** (`CUDAQVQE` with `CompoundScaleConfig`): Multi-stage compound scaling of ansatz depth/entanglement/shots
- **SubspaceVQE** (`SubspaceVQE`): Block-wise parameter optimization with alternating offsets; supports geometric or Hamiltonian-aware (graph partitioning) blocking
- **Parallel Baseline** (`CUDAQVQE` with `enable_hamiltonian_batching`): Distributes Hamiltonian terms across GPUs

### Key Modules

- `Qiskit_VQE/cudaq.py` — `CUDAQVQE` class: main CUDA-Q VQE solver with early stopping, staged training, Hamiltonian batching
- `Qiskit_VQE/subspace_vqe.py` — `SubspaceVQE`: block-wise optimizer with multi-GPU parallelism via ThreadPoolExecutor
- `Qiskit_VQE/efficientsu2.py` — Builds CUDA-Q EfficientSU2 ansatz kernels with configurable entanglement patterns
- `Qiskit_VQE/custom_ansatz.py` — Alternating-CX variant of EfficientSU2
- `Qiskit_VQE/gradient_optimizer.py` — Parameter-shift gradient computation with GPU batching, L-BFGS-B optimizer
- `Qiskit_VQE/utils_scaling.py` — `compound_scale()` maps phi index to (reps, entanglement, skip_unentangled, shots)
- `Protein_Folding/qubit_op_builder.py` — Constructs the full Hamiltonian from peptide geometry and interaction energies
- `Protein_Folding/interactions/resources/mj_matrix.txt` — Miyazawa-Jernigan energy matrix data

### Bitstring Convention
CUDA-Q uses big-endian (qubit 0 = leftmost), but the Protein_Folding interpreter expects Qiskit's little-endian (qubit 0 = rightmost). Bitstrings are reversed in `measure.py` and qubit order is reversed in `qiskit_sparsepauliop_to_cudaq_spin`. This is a critical invariant — do not change one without the other.

## QDockBank Dataset

`QDockBank_/` contains the benchmark dataset of quantum-predicted protein fragment structures for docking evaluation. Each fragment has a `.pdb` structure, `_metadata.json` (qubit count, circuit depth, energies), and `_RMSD_docking_result.json` (RMSD and docking affinities across 20 runs).

## Dependencies

Core: `cudaq`, `qiskit`, `qiskit-ibm-runtime`, `scipy`, `numpy`, `networkx` (optional, for Hamiltonian-aware blocking)

Post-processing: `modeller` (full atomic reconstruction), `openbabel` (file conversion), AutoDock Vina (docking)
