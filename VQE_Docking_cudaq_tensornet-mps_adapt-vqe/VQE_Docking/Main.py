# --*-- conding:utf-8 --*--
# @Time : 1/17/25 11:14 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Main.py

import json
import os
import sys
import time
import math
from tkinter.constants import TRUE
from typing import Dict, List, Tuple, Optional

# Ensure CUDA device visibility is set before cudaq initializes
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3,4,5,6")  # no spaces (CUDA dislikes them)

# Use editable cudaq source from /home/fengy46/VQE_Docking_cudaq/cudaq instead of installed package.
# The .so files are symlinked back to the conda env's compiled binaries.
# NOTE: LD_LIBRARY_PATH must be set BEFORE launching Python. Use run.sh or:
#   export LD_LIBRARY_PATH=~/conda/envs/cudaq-env/lib/python3.11/site-packages/lib:$LD_LIBRARY_PATH
#   PYTHONPATH=/home/fengy46/VQE_Docking_cudaq/cudaq:$PYTHONPATH python Main.py
# cudaq comes from cudaq_local via PYTHONPATH set in run.sh (do not hardcode a path here).
# sys.path.insert(0, os.path.expanduser("/home/fengy46/VQE_Docking_cudaq/cudaq"))

import cudaq
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
from qiskit_ibm_runtime import QiskitRuntimeService

from Qiskit_VQE import (
    CompoundScaleConfig,
    CUDAQVQE,
    CudaqStateCalculator,
    VQE5,
    StateCalculator,
    qiskit_sparsepauliop_to_cudaq_spin,
    SubspaceVQE,
    AdaptVQE,
)

# # Backend selection: "mgpu" for large qubits (memory pooling), "mqpu" for parallel block processing
# CUDAQ_BACKEND = "mqpu"  # Options: "mgpu" or "mqpu"

# print(f"Number of available GPUs: {cudaq.num_available_gpus()}")
# print(f"Using CUDA-Q backend: {CUDAQ_BACKEND}")
# # Append fp64 precision for mgpu backend
# cudaq_option = f"{CUDAQ_BACKEND},fp64" if CUDAQ_BACKEND == "mgpu" else CUDAQ_BACKEND
# cudaq.set_target('nvidia', option=cudaq_option)
# print(f"CUDA-Q target set to: nvidia with option={cudaq_option}")
# cudaq.mpi.initialize()
# # Determine parallelization strategy based on backend:
# #   - mgpu: Sequential block processing (MPI-based multi-GPU memory pooling for larger qubits)
# #   - mqpu: Parallel block processing (independent GPU simulators for algorithm-level parallelism)
# USE_PARALLEL_BLOCKS = (CUDAQ_BACKEND == "mqpu")

# --- MPS bond dimension cap (CUDAQ_MPS_MAX_BOND) ------------------------------
# THIS is the single place to control the bond dimension. It is a CAP: each bond
# uses the smallest chi that captures the state, capped at this value.
#   BOND_DIM = "auto"  -> set per protein to the value that reproduces a FULL
#                         STATEVECTOR EXACTLY for the active ansatz (linear
#                         EfficientSU2, reps=shared_reps): chi=2^reps=16 for
#                         reps=4 (chi=8 for the 7-qubit proteins). Recommended:
#                         exact accuracy at minimum memory.
#   BOND_DIM = <int>   -> force a fixed cap for every protein (e.g. 256, 1024).
# (Applied per protein inside predict_protein_structure; see exact_bond_dim.)
BOND_DIM = "auto"

cudaq.set_target('tensornet-mps-mgpu')
cudaq.mpi.initialize()


def exact_bond_dim(entanglement: str, reps: int, n_qubits: int) -> int:
    """Bond dimension chi that makes tensornet-mps reproduce a full statevector
    EXACTLY for an EfficientSU2 ansatz. Determined by the ansatz STRUCTURE (not
    the protein sequence): chi <= 2^(max #2-qubit gates crossing any chain cut),
    capped by the trivial Schmidt bound 2^(n/2).
        linear / reverse_linear -> 1 straddling CX per rep        -> 2^reps
        circular / sca          -> linear + wrap-around (every cut) -> 2^(2*reps)
        full / pairwise         -> all-to-all -> saturates 2^(n/2) (no MPS gain)
    Setting CUDAQ_MPS_MAX_BOND to this value guarantees statevector-level
    accuracy at the minimum memory for that ansatz.
    """
    half = max(0, n_qubits // 2)
    cap = 1 << half
    e = entanglement.lower()
    if e in ("linear", "reverse_linear"):
        need = 1 << reps
    elif e in ("circular", "sca"):
        need = 1 << (2 * reps)
    else:  # full, pairwise, ...
        need = cap
    return max(1, min(need, cap))


def predict_protein_structure(
    main_chain_sequence: str,
    protein_id: str,
    platform: str, # ibm_quantum or cudaq
    service: QiskitRuntimeService,
    max_iter: int = 150,
    efficientnet_entanglement_schedule: Optional[List[str]] = None,
):
    """
    Predicts a protein structure using a quantum VQE workflow and saves the results.

    :param main_chain_sequence: The main chain amino acid sequence (single-letter representation).
    :param protein_id: The identifier/name for the protein (used for output directories and files).
    :param service: An instance of QiskitRuntimeService for submitting quantum jobs.
    :param max_iter: Maximum iteration count for VQE optimization, default is 150.
    :param efficientnet_entanglement_schedule: Ordered list of entanglement patterns to use for EfficientNet scaling.
    """
    print(f"Starting prediction for protein: {protein_id}, sequence: {main_chain_sequence}")

    # Initialize peptide with an empty side-chain sequence
    side_chain_sequences = ['' for _ in main_chain_sequence]
    peptide = Peptide(main_chain_sequence, side_chain_sequences)

    # Define interactions and penalty parameters
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(10, 10, 10)

    # Create the protein folding problem
    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    hamiltonian = protein_folding_problem.qubit_op()

    qubit_count = hamiltonian.num_qubits
    print(f"Number of qubits required: {qubit_count}")

    if platform == "ibm_quantum":
        vqe_instance = VQE5(
            service=service,
            hamiltonian=hamiltonian,
            min_qubit_num=qubit_count,
            maxiter=max_iter,
        )
        raise NotImplementedError("IBM Quantum path needs EfficientNet-style integration.")

    if platform != "cudaq":
        raise ValueError(f"Invalid platform: {platform}")

    cudaq_hamiltonian = qiskit_sparsepauliop_to_cudaq_spin(hamiltonian)

    # ====================================================================================
    # SHARED HYPERPARAMETERS: All strategies use the same core hyperparameters
    # ====================================================================================
    # Measurement shots -- the SINGLE knob, used consistently for the VQE and the
    # bitstring readout. On tensornet-mps-mgpu the VQE energy is ALWAYS exact
    # (observe() is analytic and ignores shots), so shared_shots only controls the
    # bitstring readout: >0 = shot-based sampling (works for any qubit count, and
    # matches QDockBank's real-hardware readout); -1 = exact statevector readout
    # (only feasible below ~28 qubits, since get_state materializes 2^n).
    shared_shots = 100000
    shared_reps = 4                              # Ansatz repetition depth (= QDockBank reps)

    # NOTE: the MPS bond dimension (CUDAQ_MPS_MAX_BOND) is set PER STRATEGY inside
    # the strategy loop below, from each strategy's ACTUAL entanglement pattern +
    # reps -- so a statevector-exact chi is chosen automatically for whichever
    # ansatz is enabled (linear -> 2^reps, circular -> 2^(2*reps), full -> ~2^(n/2)).
    # It uses the real CUDA-Q qubit_count (~5 fewer than the QDockBank count).
    # See BOND_DIM at the top of the file ("auto" vs a fixed integer cap).
    shared_seed = 0                              # Random seed for reproducibility
    shared_energy_tol = 1e-4                     # Energy improvement threshold for early stopping
    shared_use_gradient_optimizer = False         # Use L-BFGS-B gradient optimizer (all strategies)

    # Derived hyperparameters (based on shots)
    shared_patience = 12  # Patience for early stopping
    shared_window = 40  # Window size for energy smoothing
    shared_max_iter = max_iter                              # Max iterations (use raw max_iter=1000000)
    USE_PARALLEL_BLOCKS = True   # SubspaceVQE parallel blocks: each block runs on its own GPU (round-robin via CUDAQ_MPS_NUM_GPUS=1)

    # EfficientNet-specific hyperparameters (multi-stage scaling)
    if efficientnet_entanglement_schedule is None:
        efficientnet_entanglement_schedule = ["circular", "circular", "full"]
    if len(efficientnet_entanglement_schedule) == 0:
        raise ValueError("efficientnet_entanglement_schedule must contain at least one entanglement pattern.")
    efficientnet_entanglement_schedule = [str(pattern) for pattern in efficientnet_entanglement_schedule]

    efficientnet_gamma = 1.0                                        # Fixed shots (no scaling with phi)
    efficientnet_r0 = shared_shots                                  # Base shots for EfficientNet
    efficientnet_min_energy_tol = 5e-5                              # Minimum energy tolerance floor
    efficientnet_energy_tol_decay = 0.8                             # Energy tolerance decay per stage
    efficientnet_phi_schedule = list(range(len(efficientnet_entanglement_schedule)))

    # ====================================================================================
    # BASELINE STRATEGIES: 6 single-stage configurations (one per entanglement pattern)
    # ====================================================================================
    baseline_strategies = {
        "baseline_full": {
            "description": "Baseline: Single-stage with full entanglement",
            "entanglement": "full",
        },
        "baseline_linear": {
            "description": "Baseline: Single-stage with linear entanglement",
            "entanglement": "linear",
        },
        "baseline_reverse_linear": {
            "description": "Baseline: Single-stage with reverse_linear entanglement",
            "entanglement": "reverse_linear",
        },
        "baseline_circular": {
            "description": "Baseline: Single-stage with circular entanglement",
            "entanglement": "circular",
        },
        "baseline_pairwise": {
            "description": "Baseline: Single-stage with pairwise entanglement",
            "entanglement": "pairwise",
        },
        "baseline_sca": {
            "description": "Baseline: Single-stage with sca entanglement",
            "entanglement": "sca",
        },
    }

    # ====================================================================================
    # EFFICIENTNET STRATEGIES: 6 multi-stage configurations
    # ====================================================================================
    efficientnet_strategies = {
        "efficientnet_reverse_linear_reverse_linear_reverse_linear": {
            "description": "EfficientNet: Homogeneous [reverse_linear, reverse_linear, reverse_linear]",
            "schedule": ["reverse_linear", "reverse_linear", "reverse_linear"],
        },
        "efficientnet_linear_linear_linear": {
            "description": "EfficientNet: Homogeneous [linear, linear, linear]",
            "schedule": ["linear", "linear", "linear"],
        },
        "efficientnet_full_full_full": {
            "description": "EfficientNet: Homogeneous [full, full, full]",
            "schedule": ["full", "full", "full"],
        },
        "efficientnet_circular_circular_circular": {
            "description": "EfficientNet: Homogeneous [circular, circular, circular]",
            "schedule": ["circular", "circular", "circular"],
        },
        "efficientnet_pairwise_pairwise_pairwise": {
            "description": "EfficientNet: Homogeneous [pairwise, pairwise, pairwise]",
            "schedule": ["pairwise", "pairwise", "pairwise"],
        },
        "efficientnet_sca_sca_sca": {
            "description": "EfficientNet: Homogeneous [sca, sca, sca]",
            "schedule": ["sca", "sca", "sca"],
        },
    }

    # Build the complete strategies dictionary
    strategies: Dict[str, Dict] = {}

    # Add all baseline strategies
    # for strategy_name, strategy_info in baseline_strategies.items():
    #     strategies[strategy_name] = {
    #         "description": strategy_info["description"],
    #         "vqe_kwargs": dict(
    #             hamiltonian=cudaq_hamiltonian,
    #             max_iterations=shared_max_iter,
    #             shots=shared_shots,
    #             seed=shared_seed,
    #             energy_tol=shared_energy_tol,
    #             patience=shared_patience,
    #             window=shared_window,
    #             reps=shared_reps,
    #             entanglement=strategy_info["entanglement"],
    #             skip_unentangled_qubits=False,
    #             ansatz_variant="efficientsu2",
    #             use_gradient_optimizer=False,
    #             adaptive_iterations=False,  # NO early stopping: run the full COBYLA budget
    #         ),
    #     }

    # ====================================================================================
    # PARALLEL BASELINE STRATEGIES: Baseline with Hamiltonian Batching (multi-GPU)
    # Uses the same approach as global refinement in SubspaceVQE - distributes
    # Hamiltonian terms across multiple GPUs for accelerated energy evaluation.
    # ====================================================================================
    parallel_baseline_strategies = {
        "parallel_baseline_full": {
            "description": "Parallel Baseline: Hamiltonian batching with full entanglement",
            "entanglement": "full",
        },
        "parallel_baseline_linear": {
            "description": "Parallel Baseline: Hamiltonian batching with linear entanglement",
            "entanglement": "linear",
        },
        "parallel_baseline_reverse_linear": {
            "description": "Parallel Baseline: Hamiltonian batching with reverse_linear entanglement",
            "entanglement": "reverse_linear",
        },
        "parallel_baseline_circular": {
            "description": "Parallel Baseline: Hamiltonian batching with circular entanglement",
            "entanglement": "circular",
        },
        "parallel_baseline_pairwise": {
            "description": "Parallel Baseline: Hamiltonian batching with pairwise entanglement",
            "entanglement": "pairwise",
        },
        "parallel_baseline_sca": {
            "description": "Parallel Baseline: Hamiltonian batching with sca entanglement",
            "entanglement": "sca",
        },
    }

    # Add all parallel baseline strategies (uses Hamiltonian batching like global refinement)
    # for strategy_name, strategy_info in parallel_baseline_strategies.items():
    #     strategies[strategy_name] = {
    #         "description": strategy_info["description"],
    #         "vqe_kwargs": dict(
    #             hamiltonian=cudaq_hamiltonian,
    #             max_iterations=shared_max_iter,
    #             shots=shared_shots,
    #             seed=shared_seed,
    #             energy_tol=shared_energy_tol,
    #             patience=shared_patience,
    #             window=shared_window,
    #             reps=shared_reps,
    #             entanglement=strategy_info["entanglement"],
    #             skip_unentangled_qubits=False,
    #             ansatz_variant="efficientsu2",
    #             use_gradient_optimizer=shared_use_gradient_optimizer,
    #             enable_hamiltonian_batching=USE_PARALLEL_BLOCKS,  # Enable multi-GPU Hamiltonian batching
    #         ),
    #     }

    # Add all EfficientNet strategies
    # for strategy_name, strategy_info in efficientnet_strategies.items():
    #     strategies[strategy_name] = {
    #         "description": strategy_info["description"],
    #         "vqe_kwargs": dict(
    #             hamiltonian=cudaq_hamiltonian,
    #             max_iterations=shared_max_iter,
    #             shots=shared_shots,
    #             seed=shared_seed,
    #             energy_tol=shared_energy_tol,
    #             patience=shared_patience,
    #             window=shared_window,
    #             ansatz_variant="efficientsu2",
    #             phi_schedule=efficientnet_phi_schedule,
    #             scale_config=CompoundScaleConfig(
    #                 d0=2,
    #                 r0=efficientnet_r0,
    #                 alpha=1.5,
    #                 gamma=efficientnet_gamma,
    #                 entanglement_schedule=tuple(strategy_info["schedule"]),
    #                 skip_unentangled_phi_threshold=-1.0,  # Never skip: consistent circuit structure across all stages
    #             ),
    #             energy_tol_decay=efficientnet_energy_tol_decay,
    #             min_energy_tol=efficientnet_min_energy_tol,
    #             tol_scale=0.02,
    #             use_gradient_optimizer=shared_use_gradient_optimizer,
    #             enable_hamiltonian_batching=USE_PARALLEL_BLOCKS,  # Distribute Hamiltonian terms across GPUs per stage
    #         ),
    #     }

    # ====================================================================================
    # SUBSPACE VQE STRATEGIES: Block-wise optimization with alternating offsets
    # ====================================================================================
    subspace_strategies = {
        "subspace_window2_circular": {
            "description": "SubspaceVQE: Block-wise (size=2) with alternating offsets, circular entanglement",
            "entanglement": "circular",
            "window_size": 2,
        },
        "subspace_window2_full": {
            "description": "SubspaceVQE: Block-wise (size=2) with alternating offsets, full entanglement",
            "entanglement": "full",
            "window_size": 2,
        },
        "subspace_window2_linear": {
            "description": "SubspaceVQE: Block-wise (size=2) with alternating offsets, linear entanglement",
            "entanglement": "linear",
            "window_size": 2,
        },
        "subspace_window2_reverse_linear": {
            "description": "SubspaceVQE: Block-wise (size=2) with alternating offsets, reverse_linear entanglement",
            "entanglement": "reverse_linear",
            "window_size": 2,
        },
        "subspace_window2_pairwise": {
            "description": "SubspaceVQE: Block-wise (size=2) with alternating offsets, pairwise entanglement",
            "entanglement": "pairwise",
            "window_size": 2,
        },
        "subspace_window2_sca": {
            "description": "SubspaceVQE: Block-wise (size=2) with alternating offsets, sca entanglement",
            "entanglement": "sca",
            "window_size": 2,
        },
    }

    # Add all SubspaceVQE strategies (geometric blocking)
    # for strategy_name, strategy_info in subspace_strategies.items():
    #     strategies[strategy_name] = {
    #         "description": strategy_info["description"],
    #         "strategy_type": "subspace",  # Mark as subspace strategy
    #         "vqe_kwargs": dict(
    #             hamiltonian=cudaq_hamiltonian,
    #             num_qubits=qubit_count,
    #             shots=shared_shots,
    #             max_epochs=shared_max_iter,
    #             window_size=strategy_info["window_size"],
    #             block_maxiter=shared_max_iter,
    #             global_refinement=True,
    #             global_maxiter=shared_max_iter,
    #             reps=shared_reps,
    #             entanglement=strategy_info["entanglement"],
    #             # Early stopping parameters (same as all strategies)
    #             energy_tol=shared_energy_tol,
    #             patience=shared_patience,
    #             epoch_patience=shared_patience,  # Use same patience for epoch-level
    #             window=shared_window,
    #             seed=shared_seed,
    #             enable_parallel=USE_PARALLEL_BLOCKS,
    #             max_parallel_blocks=None if USE_PARALLEL_BLOCKS else 1,
    #             force_sequential=not USE_PARALLEL_BLOCKS,
    #             use_hamiltonian_blocking=False,  # Geometric blocking
    #             use_gradient_optimizer=shared_use_gradient_optimizer,
    #         ),
    #     }

    # ====================================================================================
    # HAMILTONIAN-AWARE SUBSPACE VQE: Graph partitioning-based block optimization
    # ====================================================================================
    for _ent in ["circular", "linear", "reverse_linear", "sca"]:
        strategies[f"adapt_vqe_{_ent}"] = {
            "description": f"ADAPT-VQE (gradient_threshold=1e-3, COBYLA) with {_ent} entanglement",
            "strategy_type": "adapt",
            "vqe_class": AdaptVQE,
            "vqe_kwargs": dict(
                hamiltonian=cudaq_hamiltonian,
                num_qubits=qubit_count,
                entanglement=_ent,
                gradient_threshold=1e-3,
                max_iterations=None,
                max_iter=shared_max_iter,
                energy_tol=shared_energy_tol,
                patience=shared_patience,
                window=shared_window,
                shots=shared_shots,
                seed=shared_seed,
            ),
        }

    performance_records: List[Dict[str, float]] = []

    for strategy_name, strategy_cfg in strategies.items():
        # Skip if this (protein, strategy) pair already has a completed energy file.
        _energy_file_check = os.path.join(
            "Result", "process_data", strategy_name, protein_id,
            "System_Energy", f"energy_list_{protein_id}.txt"
        )
        if os.path.isfile(_energy_file_check) and os.path.getsize(_energy_file_check) > 0:
            print(f"[{protein_id}] Skipping '{strategy_name}' — already completed ({_energy_file_check})")
            continue

        print(f"[{protein_id}] Running strategy '{strategy_name}' ({strategy_cfg['description']})")
        vqe_kwargs = strategy_cfg["vqe_kwargs"]

        # Statevector-EXACT bond dimension for THIS strategy's ansatz, set before
        # the VQE runs (the backend reads CUDAQ_MPS_MAX_BOND per circuit). Adapts
        # automatically to the strategy's entanglement pattern + reps:
        #   linear/reverse_linear -> 2^reps,  circular/sca -> 2^(2*reps),
        #   full/pairwise -> ~2^(n/2) (MPS no longer efficient there).
        # For multi-stage scaling (efficientnet) this uses the base reps/pattern;
        # set BOND_DIM to a fixed integer to override.
        if BOND_DIM == "auto":
            _ent = str(vqe_kwargs.get("entanglement", "linear"))
            _reps = int(vqe_kwargs.get("reps", shared_reps))
            _chi = exact_bond_dim(_ent, _reps, qubit_count)
            _tag = "statevector-exact"
        else:
            _ent = "fixed"; _reps = int(vqe_kwargs.get("reps", shared_reps))
            _chi = int(BOND_DIM); _tag = "manual cap"
        os.environ["CUDAQ_MPS_MAX_BOND"] = str(_chi)
        print(f"[{protein_id}] {strategy_name}: ansatz='{_ent}', reps={_reps}, "
              f"n={qubit_count} -> CUDAQ_MPS_MAX_BOND={_chi} ({_tag})")

        # Choose VQE class based on strategy type
        vqe_class = strategy_cfg.get("vqe_class")
        if vqe_class is not None:
            vqe_instance = vqe_class(**vqe_kwargs)
        elif strategy_cfg.get("strategy_type") == "subspace":
            vqe_instance = SubspaceVQE(**vqe_kwargs)
        else:
            vqe_instance = CUDAQVQE(**vqe_kwargs)

        start = time.time()
        energy_list, final_params, ansatz, top_results = vqe_instance.run_vqe()
        elapsed = time.time() - start

        strategy_root = os.path.join("Result", "process_data", strategy_name, protein_id)
        energy_dir = os.path.join(strategy_root, "System_Energy")
        prob_dir = os.path.join(strategy_root, "Prob_distribution")
        structure_dir = os.path.join(strategy_root, "Structures")
        os.makedirs(energy_dir, exist_ok=True)
        os.makedirs(prob_dir, exist_ok=True)
        os.makedirs(structure_dir, exist_ok=True)

        energy_file = os.path.join(energy_dir, f"energy_list_{protein_id}.txt")
        with open(energy_file, "w") as file:
            file.writelines(f"{energy}\n" for energy in energy_list)

        # Build metadata (different fields for different VQE types)
        metadata = {
            "strategy": strategy_name,
            "description": strategy_cfg["description"],
            "iterations": len(energy_list),
            "final_shots": vqe_instance.shots,
            "execution_time_s": elapsed,
        }

        # Add strategy-specific metadata
        if strategy_cfg.get("strategy_type") == "subspace":
            # SubspaceVQE-specific fields
            metadata.update({
                "strategy_type": "subspace",
                "window_size": vqe_instance.window_size,
                "max_epochs": vqe_instance.max_epochs,
                "epoch_summaries": vqe_instance.epoch_summaries,
                "num_function_evaluations": vqe_instance.nfev,
            })
        else:
            # CUDAQVQE-specific fields
            metadata.update({
                "strategy_type": "standard",
                "phi_schedule": [stage.get("phi") for stage in vqe_instance.stage_plan],
                "stage_offsets": vqe_instance.stage_offsets,
                "stage_summaries": vqe_instance.stage_summaries,
                "adaptive_iterations": vqe_instance.use_adaptive_iterations,
            })
        meta_path = os.path.join(strategy_root, "stage_summary.json")
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        # Probability distribution and derived structures. Readout uses the same
        # shared_shots knob: >0 -> sampling (any qubit count; matches QDockBank
        # hardware), -1 -> exact statevector (small n only). The VQE energy above
        # is exact regardless (observe is analytic on this backend).
        state_calculator = CudaqStateCalculator(ansatz, shots=shared_shots)

        probability_distribution = state_calculator.get_probability_distribution(final_params)

        prob_file = os.path.join(prob_dir, f"prob_distribution_{protein_id}.txt")
        with open(prob_file, "w") as file:
            for state, probability in probability_distribution.items():
                file.write(f"{state}: {probability}\n")

        protein_result = protein_folding_problem.interpret(probability_distribution)
        protein_result.save_xyz_file(
            name=f"{protein_id}_{strategy_name}",
            path=structure_dir,
        )
        print(f"[{protein_id}] Strategy '{strategy_name}' final structure saved.")

        for rank, (energy_val, best_params) in enumerate(top_results, start=1):
            print(f"[{protein_id}] {strategy_name} Top {rank} energy = {energy_val}")
            best_prob_dist = state_calculator.get_probability_distribution(best_params)
            best_protein_result = protein_folding_problem.interpret(best_prob_dist)
            best_protein_result.save_xyz_file(
                name=f"{protein_id}_{strategy_name}_top_{rank}",
                path=structure_dir,
            )

        # Append execution info for logging
        yield_data = {
            "protein_id": protein_id,
            "strategy": strategy_name,
            "elapsed": elapsed,
        }
        performance_records.append(yield_data)

    return performance_records

def main():
    """Main function to predict protein structures for a list of proteins."""
    platform = 'cudaq'
    # Initialize the Qiskit Runtime Service
    if platform == 'ibm_quantum':
        service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance='YOUR_INSTANCE_NAME',  # Replace with your real instance
            token='YOUR_API_TOKEN'          # Replace with your real token
        )
    elif platform == 'cudaq':
        service = None
    else:
        raise ValueError(f"Invalid platform: {platform}")

    # List of proteins to process
    # protein_list: List[Tuple[str, str]] = [
    #     ("DGKMKGLAF", "1qin"),
    #     ("IHGIGGFI", "1a9m"),
    #     ("KSIVDSGTTNLR", "1fkn"),
    #     ("NNLGTIAKSGT", "3b26"),
    #     ("GAVEDGATMTFF", "2xxx"),
    #     ("DWGGM", "3ans"),
    #     ("YAGYS", "6mu3")
    # ]

    # All 55 QDockBank proteins. NOTE: QDockBank qubit counts are from REAL
    # hardware and include ~5 extra qubits that CUDA-Q does not need; the
    # actual CUDA-Q qubit count (hamiltonian.num_qubits) is smaller, and the
    # per-protein statevector-exact bond dimension is computed from it.
    protein_list: List[Tuple[str, str]] = [
        ('VKDRS',          '3ckz'),     # 5 res, QDockBank 12q/depth 53
        ('RYRDV',          '3eax'),     # 5 res, QDockBank 12q/depth 53
        ('NIGGF',          '4mo4'),     # 5 res, QDockBank 12q/depth 53
        ('DGPHGM',         '1e2k'),     # 6 res, QDockBank 23q/depth 97
        ('SIHSYS',         '1hdq'),     # 6 res, QDockBank 23q/depth 97
        ('ATFTIT',         '2v25'),     # 6 res, QDockBank 23q/depth 97
        ('IQFHFH',         '3ibi'),     # 6 res, QDockBank 23q/depth 97
        ('GIKAVM',         '3s0b'),     # 6 res, QDockBank 23q/depth 97
        ('LRKANG',         '6czf'),     # 6 res, QDockBank 23q/depth 97
        ('QQTMLRV',        '1qin'),     # 7 res, QDockBank 38q/depth 157
        ('HNDPGWI',        '3dx3'),     # 7 res, QDockBank 38q/depth 157
        ('CDLCSVT',        '5c28'),     # 7 res, QDockBank 38q/depth 157
        ('PWWERYQP',       '1ppi'),     # 8 res, QDockBank 46q/depth 189
        ('VCHASAWD',       '3nxq'),     # 8 res, QDockBank 46q/depth 189
        ('IEGVPESN',       '3tcg'),     # 8 res, QDockBank 46q/depth 189
        ('SLTTPPLL',       '4q87'),     # 8 res, QDockBank 46q/depth 189
        ('GSYSDVSI',       '4xaq'),     # 8 res, QDockBank 46q/depth 189
        ('GGPNGWKV',       '4zb8'),     # 8 res, QDockBank 46q/depth 189
        ('SLTTPPLL',       '5tya'),     # 8 res, QDockBank 46q/depth 189
        ('VYSSGIPL',       '6p86'),     # 8 res, QDockBank 46q/depth 189
        ('AQITMGMPY',      '1e2l'),     # 9 res, QDockBank 54q/depth 221
        ('SAPLRVYVE',      '1gx8'),     # 9 res, QDockBank 54q/depth 221
        ('TAGATSANE',      '1m7y'),     # 9 res, QDockBank 54q/depth 221
        ('DACQGDSGG',      '4y79'),     # 9 res, QDockBank 54q/depth 221
        ('LLDTGADDTV',     '1zsf'),     # 10 res, QDockBank 63q/depth 257
        ('LIDTGADDTV',     '2avo'),     # 10 res, QDockBank 63q/depth 257
        ('EDACQGDSGG',     '2bok'),     # 10 res, QDockBank 63q/depth 257
        ('EDACQGDSGG',     '2vwo'),     # 10 res, QDockBank 63q/depth 257
        ('ELISNSSDAL',     '3b26'),     # 10 res, QDockBank 63q/depth 257
        ('YLVTHLMGAD',     '3d83'),     # 10 res, QDockBank 63q/depth 257
        ('LLDTGADDTV',     '3vf7'),     # 10 res, QDockBank 63q/depth 257
        ('LLDTGADDTV',     '4mc1'),     # 10 res, QDockBank 63q/depth 257
        ('LLNTGADDTV',     '5kqx'),     # 10 res, QDockBank 63q/depth 257
        ('LLNTGADDTV',     '5kr2'),     # 10 res, QDockBank 63q/depth 257
        ('AKQRLKCASL',     '6ezq'),     # 10 res, QDockBank 63q/depth 257
        ('HCSAGIGRSGT',    '2qbs'),     # 11 res, QDockBank 72q/depth 293
        ('GLAWSYYIGYL',    '4f5y'),     # 11 res, QDockBank 72q/depth 293
        ('FDGKGGILAHA',    '5cxa'),     # 11 res, QDockBank 72q/depth 293
        ('MIITEYMENGA',    '5nkd'),     # 11 res, QDockBank 72q/depth 293
        ('RNNGHSVQLTL',    '6g98'),     # 11 res, QDockBank 72q/depth 293
        ('AFPAVSAGIYGC',   '2bfq'),     # 12 res, QDockBank 82q/depth 333
        ('GAVEDGATMTFF',   '2xxx'),     # 12 res, QDockBank 82q/depth 333
        ('MIITEYMENGAL',   '5nkc'),     # 12 res, QDockBank 82q/depth 333
        ('ELISNSSDALDKI',  '1yc4'),     # 13 res, QDockBank 92q/depth 373
        ('DYLEAYGKGGVKA',  '4jpx'),     # 13 res, QDockBank 92q/depth 373
        ('RKLGRGKYSEVFE',  '5cqu'),     # 13 res, QDockBank 92q/depth 373
        ('YLVTHLMGADLNNI', '3d7z'),     # 14 res, QDockBank 102q/depth 413
        ('VVLPYMKHGDLRNF', '4aoi'),     # 14 res, QDockBank 102q/depth 413
        ('VRDQAEHLKTAVQM', '4cig'),     # 14 res, QDockBank 102q/depth 413
        ('ILMELMAGGDLKSF', '4clj'),     # 14 res, QDockBank 102q/depth 413
        ('PVHTAVGTVGTAPL', '4fp1'),     # 14 res, QDockBank 102q/depth 413
        ('DYLEAYGKGGVKAK', '4jpy'),     # 14 res, QDockBank 102q/depth 413
        ('IEGLEGAGKTTARN', '4tmk'),     # 14 res, QDockBank 102q/depth 413
        ('MIITEYMENGALDK', '5nkb'),     # 14 res, QDockBank 102q/depth 413
        ('SLSRVMIHVFSDGV', '6udv'),     # 14 res, QDockBank 102q/depth 413
    ]

    # Edit this list to choose the three entanglement patterns and their order.
    efficientnet_entanglement_schedule: List[str] = [
        "full",
        "full",
        "full",
    ]


    # Log execution times for each protein
    log_file_path = "execution_time_log.txt"
    with open(log_file_path, 'w') as log_file:
        log_file.write("Protein_ID\tStrategy\tExecution_Time(s)\n")

        for sequence, protein_name in protein_list:
            records = predict_protein_structure(
                main_chain_sequence=sequence,
                protein_id=protein_name,
                platform=platform,
                service=service,
                max_iter=1000000,  # No hard cap; early stopping drives convergence
                efficientnet_entanglement_schedule=efficientnet_entanglement_schedule,
            )

            for record in records:
                log_file.write(f"{record['protein_id']}\t{record['strategy']}\t{record['elapsed']:.2f}\n")

if __name__ == '__main__':
    main()
    cudaq.mpi.finalize()
