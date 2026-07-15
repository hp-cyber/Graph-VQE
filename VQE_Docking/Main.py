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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2, 3, 4, 5")

# Use editable cudaq source from /home/fengy46/VQE_Docking_cudaq/cudaq instead of installed package.
# The .so files are symlinked back to the conda env's compiled binaries.
# NOTE: LD_LIBRARY_PATH must be set BEFORE launching Python. Use run.sh or:
#   export LD_LIBRARY_PATH=~/conda/envs/cudaq-env/lib/python3.11/site-packages/lib:$LD_LIBRARY_PATH
#   PYTHONPATH=/home/fengy46/VQE_Docking_cudaq/cudaq:$PYTHONPATH python Main.py
sys.path.insert(0, os.path.expanduser("/home/fengy46/VQE_Docking_cudaq/cudaq"))

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
)

# Backend selection: "mgpu" for large qubits (memory pooling), "mqpu" for parallel block processing
CUDAQ_BACKEND = "mqpu"  # Options: "mgpu" or "mqpu"

print(f"Number of available GPUs: {cudaq.num_available_gpus()}")
print(f"Using CUDA-Q backend: {CUDAQ_BACKEND}")
# Append fp64 precision for mgpu backend
cudaq_option = f"{CUDAQ_BACKEND},fp64" if CUDAQ_BACKEND == "mgpu" else CUDAQ_BACKEND
cudaq.set_target('nvidia', option=cudaq_option)
print(f"CUDA-Q target set to: nvidia with option={cudaq_option}")
cudaq.mpi.initialize()
# Determine parallelization strategy based on backend:
#   - mgpu: Sequential block processing (MPI-based multi-GPU memory pooling for larger qubits)
#   - mqpu: Parallel block processing (independent GPU simulators for algorithm-level parallelism)
USE_PARALLEL_BLOCKS = (CUDAQ_BACKEND == "mqpu")

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
    shared_shots = 8192                          # Measurement shots (consistent across all strategies)
    shared_reps = 4                              # Ansatz repetition depth
    shared_seed = 0                              # Random seed for reproducibility
    shared_energy_tol = 1e-4                     # Energy improvement threshold for early stopping
    shared_use_gradient_optimizer = True         # Use L-BFGS-B gradient optimizer (all strategies)

    # Derived hyperparameters (based on shots)
    shared_patience = 12  # Patience for early stopping
    shared_window = 40  # Window size for energy smoothing
    shared_max_iter = max_iter                              # Max iterations (use raw max_iter=1000000)

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
    hamiltonian_aware_strategies = {
        "subspace_hamiltonian_circular": {
            "description": "SubspaceVQE: Hamiltonian-aware blocking (graph partitioning), circular entanglement",
            "entanglement": "circular",
        },
        "subspace_hamiltonian_full": {
            "description": "SubspaceVQE: Hamiltonian-aware blocking (graph partitioning), full entanglement",
            "entanglement": "full",
        },
        "subspace_hamiltonian_linear": {
            "description": "SubspaceVQE: Hamiltonian-aware blocking (graph partitioning), linear entanglement",
            "entanglement": "linear",
        },
        "subspace_hamiltonian_reverse_linear": {
            "description": "SubspaceVQE: Hamiltonian-aware blocking (graph partitioning), reverse_linear entanglement",
            "entanglement": "reverse_linear",
        },
        "subspace_hamiltonian_pairwise": {
            "description": "SubspaceVQE: Hamiltonian-aware blocking (graph partitioning), pairwise entanglement",
            "entanglement": "pairwise",
        },
        "subspace_hamiltonian_sca": {
            "description": "SubspaceVQE: Hamiltonian-aware blocking (graph partitioning), sca entanglement",
            "entanglement": "sca",
        },
    }

    # Add all Hamiltonian-aware SubspaceVQE strategies
    for strategy_name, strategy_info in hamiltonian_aware_strategies.items():
        strategies[strategy_name] = {
            "description": strategy_info["description"],
            "strategy_type": "subspace",  # Mark as subspace strategy
            "vqe_kwargs": dict(
                hamiltonian=cudaq_hamiltonian,
                num_qubits=qubit_count,
                shots=shared_shots,
                max_epochs=shared_max_iter,
                window_size=2,  # Not used for Hamiltonian-aware, but required parameter
                block_maxiter=shared_max_iter,
                global_refinement=True,
                global_maxiter=shared_max_iter,
                reps=shared_reps,
                entanglement=strategy_info["entanglement"],
                # Early stopping parameters (same as all strategies)
                energy_tol=shared_energy_tol,
                patience=shared_patience,
                epoch_patience=3,  # Use same patience for epoch-level
                window=shared_window,
                seed=shared_seed,
                enable_parallel=USE_PARALLEL_BLOCKS,
                max_parallel_blocks=None if USE_PARALLEL_BLOCKS else 1,
                force_sequential=not USE_PARALLEL_BLOCKS,
                # Hamiltonian-aware blocking parameters
                use_hamiltonian_blocking=True,  # Enable graph partitioning
                min_block_size=1,               # Minimum qubits per block
                use_gradient_optimizer=shared_use_gradient_optimizer,
            ),
        }

    performance_records: List[Dict[str, float]] = []

    for strategy_name, strategy_cfg in strategies.items():
        print(f"[{protein_id}] Running strategy '{strategy_name}' ({strategy_cfg['description']})")
        vqe_kwargs = strategy_cfg["vqe_kwargs"]

        # Choose VQE class based on strategy type
        if strategy_cfg.get("strategy_type") == "subspace":
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

        # Probability distribution and derived structures
        if vqe_instance.shots == -1:
            state_calculator = CudaqStateCalculator(ansatz, shots=-1)
        else:
            state_calculator = CudaqStateCalculator(ansatz, shots=vqe_instance.shots)

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

    protein_list: List[Tuple[str, str]] = [
        ("VKDRS", "3ckz"),
        ("RYRDV", "3eax"),
        ("NIGGF", "4mo4"),
        ("KPFKF", "1fkf"),
        ("DGPHGM", "1e2k"),
        ("ATFTIT", "2v25"),
        ("IQFHFH", "3ibi"),
        ("GIKAVM", "3s0b"),
        ("LRKANG", "6czf"),
        ("SIHSYS", "1hdq"),
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
                max_iter=1000000,  # Reasonable max iterations for SubspaceVQE blocks and global refinement
                efficientnet_entanglement_schedule=efficientnet_entanglement_schedule,
            )

            for record in records:
                log_file.write(f"{record['protein_id']}\t{record['strategy']}\t{record['elapsed']:.2f}\n")

if __name__ == '__main__':
    main()
    cudaq.mpi.finalize()
