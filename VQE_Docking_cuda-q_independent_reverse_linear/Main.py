# --*-- conding:utf-8 --*--
# @Time : 1/17/25 11:14 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Main.py

import json
import csv
import os
import time
import math
from case_completion import is_complete, mark_complete
from tkinter.constants import TRUE
from typing import Dict, List, Tuple, Optional

# Ensure CUDA device visibility is set before cudaq initializes
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

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
    IndependentSubspaceVQE,
)

from Qiskit_VQE.mps_config import TARGET, configure_full_bond, configure_gpu_groups

num_gpus = cudaq.num_available_gpus()
os.environ["CUDAQ_MQPU_NGPUS"] = str(num_gpus)
os.environ["CUDAQ_MPS_NUM_GPUS"] = "1"
os.environ["CUDAQ_MPS_HBATCH_NGPUS"] = str(num_gpus)
configure_gpu_groups(num_gpus)
try:
    cudaq.set_target(TARGET)
except RuntimeError as exc:
    raise RuntimeError(f"Launch with ./run.sh to load the custom {TARGET} build") from exc
cudaq.set_random_seed(0)
print(f"CUDA-Q target: {TARGET}; visible GPUs: {num_gpus}")
USE_PARALLEL_BLOCKS = num_gpus > 1

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
    shared_shots = 1000                          # Finite-shot measurement noise
    shared_reps = 4                              # Ansatz repetition depth
    shared_seed = 0                              # Random seed for reproducibility
    shared_energy_tol = 1.0                      # Stop when the next energy improvement is below 1
    shared_use_gradient_optimizer = False        # Use COBYLA for local and global optimization

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
    # HAMILTONIAN-AWARE SUBSPACE VQE: Graph partitioning-based block optimization
    # ====================================================================================
    hamiltonian_aware_strategies = {
        # "subspace_hamiltonian_circular": {
        #     "description": "Independent local VQEs + full-H global refinement, circular entanglement",
        #     "entanglement": "circular",
        # },
        # "subspace_hamiltonian_linear": {
        #     "description": "Independent local VQEs + full-H global refinement, linear entanglement",
        #     "entanglement": "linear",
        # },
        "subspace_hamiltonian_reverse_linear": {
            "description": "Independent local VQEs + full-H global refinement, reverse_linear entanglement",
            "entanglement": "reverse_linear",
        },
        # "subspace_hamiltonian_sca": {
        #     "description": "Independent local VQEs + full-H global refinement, sca entanglement",
        #     "entanglement": "sca",
        # },
    }

    # Add all Hamiltonian-aware SubspaceVQE strategies
    for strategy_name, strategy_info in hamiltonian_aware_strategies.items():
        strategies[strategy_name] = {
            "description": strategy_info["description"],
            "strategy_type": "subspace",  # Mark as subspace strategy
            "vqe_kwargs": dict(
                hamiltonian=cudaq_hamiltonian,
                qiskit_hamiltonian=hamiltonian,
                max_block_qubits=12,  # Local workload only; no total-qubit ceiling.
                num_qubits=qubit_count,
                shots=shared_shots,
                max_epochs=1,  # One global optimization after all local solves.
                block_maxiter=shared_max_iter,
                global_refinement=True,
                global_maxiter=shared_max_iter,
                reps=shared_reps,
                entanglement=strategy_info["entanglement"],
                # One comparison at each optimization level; no patience/window.
                energy_tol=shared_energy_tol,
                immediate_early_stopping=True,
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
        strategy_root = os.path.join("Result", "process_data", strategy_name, protein_id)
        if is_complete(strategy_root, main_chain_sequence, protein_id, strategy_name):
            print(f"[{protein_id}] Skipping completed strategy '{strategy_name}'", flush=True)
            continue
        try:
            os.remove(os.path.join(strategy_root, "COMPLETED.json"))
        except FileNotFoundError:
            pass
        print(f"[{protein_id}] Running strategy '{strategy_name}' ({strategy_cfg['description']})")
        vqe_kwargs = dict(strategy_cfg["vqe_kwargs"])
        configure_gpu_groups(num_gpus)

        # Choose VQE class based on strategy type
        if strategy_cfg.get("strategy_type") == "subspace":
            vqe_kwargs["timing_directory"] = os.path.join(strategy_root, "System_Energy")
            vqe_instance = IndependentSubspaceVQE(**vqe_kwargs)
        else:
            vqe_instance = CUDAQVQE(**vqe_kwargs)

        bond_stages = getattr(vqe_instance, "stage_plan", None)
        if bond_stages is None:
            bond_stages = [{"entanglement": vqe_instance.entanglement,
                            "reps": vqe_instance.reps}]
        full_bond = configure_full_bond(qubit_count, bond_stages)
        print(f"Circuit MPS bond cap: {full_bond}; absolute/relative cutoffs: 0")

        start = time.perf_counter()
        energy_list, final_params, ansatz, top_results = vqe_instance.run_vqe()
        elapsed = time.perf_counter() - start

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
            "completion_protocol": 1,
            "sequence": main_chain_sequence,
            "protein_id": protein_id,
            "top_results_count": len(top_results),
            "cudaq_target": TARGET,
            "mps_max_bond": full_bond,
            "mps_abs_cutoff": 0.0,
            "mps_relative_cutoff": 0.0,
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
                "strategy_type": "independent_hamiltonian_subspace",
                "partition": vqe_instance.partition_summary,
                "local_results": vqe_instance.local_results,
                "global_results": vqe_instance.global_results,
                "warm_start_energy": vqe_instance.warm_start_energy,
                "entanglement": vqe_instance.entanglement,
                "energy_tol": vqe_instance.energy_tol,
                "early_stopping": "immediate_energy_improvement",
                "measurement_noise": "finite_shot_sampling",
                "patience": None,
                "smoothing_window": None,
                "max_epochs": vqe_instance.max_epochs,
                "epoch_summaries": vqe_instance.epoch_summaries,
                "num_function_evaluations": vqe_instance.nfev,
                "energy_evaluation_total_time_s": vqe_instance._evaluation_seconds,
                "timing_files": {
                    "energy_evaluations": "System_Energy/energy_evaluation_timings.csv",
                    "optimizer_iterations": "System_Energy/iteration_timings.csv",
                    "epochs": "System_Energy/epoch_timings.csv",
                },
            })
            with open(os.path.join(energy_dir, "epoch_timings.csv"), "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "epoch", "offset", "energy_current", "energy_refined", "best_energy",
                    "duration_s", "energy_evaluation_count", "energy_evaluation_time_s",
                ])
                writer.writeheader()
                writer.writerows(vqe_instance.epoch_summaries)
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

        mark_complete(strategy_root, main_chain_sequence, protein_id, strategy_name)

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
        ("VKDRS",          "3ckz"),  # 5 res, QDockBank 12q/depth 53
        ("RYRDV",          "3eax"),  # 5 res, QDockBank 12q/depth 53
        ("NIGGF",          "4mo4"),  # 5 res, QDockBank 12q/depth 53
        ("DGPHGM",         "1e2k"),  # 6 res, QDockBank 23q/depth 97
        ("SIHSYS",         "1hdq"),  # 6 res, QDockBank 23q/depth 97
        ("ATFTIT",         "2v25"),  # 6 res, QDockBank 23q/depth 97
        ("IQFHFH",         "3ibi"),  # 6 res, QDockBank 23q/depth 97
        ("GIKAVM",         "3s0b"),  # 6 res, QDockBank 23q/depth 97
        ("LRKANG",         "6czf"),  # 6 res, QDockBank 23q/depth 97
        ("QQTMLRV",        "1qin"),  # 7 res, QDockBank 38q/depth 157
        ("HNDPGWI",        "3dx3"),  # 7 res, QDockBank 38q/depth 157
        ("CDLCSVT",        "5c28"),  # 7 res, QDockBank 38q/depth 157
        ("PWWERYQP",       "1ppi"),  # 8 res, QDockBank 46q/depth 189
        ("VCHASAWD",       "3nxq"),  # 8 res, QDockBank 46q/depth 189
        ("IEGVPESN",       "3tcg"),  # 8 res, QDockBank 46q/depth 189
        ("SLTTPPLL",       "4q87"),  # 8 res, QDockBank 46q/depth 189
        ("GSYSDVSI",       "4xaq"),  # 8 res, QDockBank 46q/depth 189
        ("GGPNGWKV",       "4zb8"),  # 8 res, QDockBank 46q/depth 189
        ("SLTTPPLL",       "5tya"),  # 8 res, QDockBank 46q/depth 189
        ("VYSSGIPL",       "6p86"),  # 8 res, QDockBank 46q/depth 189
        ("AQITMGMPY",      "1e2l"),  # 9 res, QDockBank 54q/depth 221
        ("SAPLRVYVE",      "1gx8"),  # 9 res, QDockBank 54q/depth 221
        ("TAGATSANE",      "1m7y"),  # 9 res, QDockBank 54q/depth 221
        ("DACQGDSGG",      "4y79"),  # 9 res, QDockBank 54q/depth 221
        ("LLDTGADDTV",     "1zsf"),  # 10 res, QDockBank 63q/depth 257
        ("LIDTGADDTV",     "2avo"),  # 10 res, QDockBank 63q/depth 257
        ("EDACQGDSGG",     "2bok"),  # 10 res, QDockBank 63q/depth 257
        ("EDACQGDSGG",     "2vwo"),  # 10 res, QDockBank 63q/depth 257
        ("ELISNSSDAL",     "3b26"),  # 10 res, QDockBank 63q/depth 257
        ("YLVTHLMGAD",     "3d83"),  # 10 res, QDockBank 63q/depth 257
        ("LLDTGADDTV",     "3vf7"),  # 10 res, QDockBank 63q/depth 257
        ("LLDTGADDTV",     "4mc1"),  # 10 res, QDockBank 63q/depth 257
        ("LLNTGADDTV",     "5kqx"),  # 10 res, QDockBank 63q/depth 257
        ("LLNTGADDTV",     "5kr2"),  # 10 res, QDockBank 63q/depth 257
        ("AKQRLKCASL",     "6ezq"),  # 10 res, QDockBank 63q/depth 257
        ("HCSAGIGRSGT",    "2qbs"),  # 11 res, QDockBank 72q/depth 293
        ("GLAWSYYIGYL",    "4f5y"),  # 11 res, QDockBank 72q/depth 293
        ("FDGKGGILAHA",    "5cxa"),  # 11 res, QDockBank 72q/depth 293
        ("MIITEYMENGA",    "5nkd"),  # 11 res, QDockBank 72q/depth 293
        ("RNNGHSVQLTL",    "6g98"),  # 11 res, QDockBank 72q/depth 293
        ("AFPAVSAGIYGC",   "2bfq"),  # 12 res, QDockBank 82q/depth 333
        ("GAVEDGATMTFF",   "2xxx"),  # 12 res, QDockBank 82q/depth 333
        ("MIITEYMENGAL",   "5nkc"),  # 12 res, QDockBank 82q/depth 333
        ("ELISNSSDALDKI",  "1yc4"),  # 13 res, QDockBank 92q/depth 373
        ("DYLEAYGKGGVKA",  "4jpx"),  # 13 res, QDockBank 92q/depth 373
        ("RKLGRGKYSEVFE",  "5cqu"),  # 13 res, QDockBank 92q/depth 373
        ("YLVTHLMGADLNNI", "3d7z"),  # 14 res, QDockBank 102q/depth 413
        ("VVLPYMKHGDLRNF", "4aoi"),  # 14 res, QDockBank 102q/depth 413
        ("VRDQAEHLKTAVQM", "4cig"),  # 14 res, QDockBank 102q/depth 413
        ("ILMELMAGGDLKSF", "4clj"),  # 14 res, QDockBank 102q/depth 413
        ("PVHTAVGTVGTAPL", "4fp1"),  # 14 res, QDockBank 102q/depth 413
        ("DYLEAYGKGGVKAK", "4jpy"),    # 14 res, QDockBank 102q/depth 413
        ("IEGLEGAGKTTARN", "4tmk"),  # 14 res, QDockBank 102q/depth 413
        ("MIITEYMENGALDK", "5nkb"),  # 14 res, QDockBank 102q/depth 413
        ("SLSRVMIHVFSDGV", "6udv"),  # 14 res, QDockBank 102q/depth 413
    ]

    # Edit this list to choose the three entanglement patterns and their order.
    efficientnet_entanglement_schedule: List[str] = [
        "full",
        "full",
        "full",
    ]


    # Log execution times for each protein
    log_file_path = "execution_time_log.txt"
    with open(log_file_path, 'a') as log_file:
        if log_file.tell() == 0:
            log_file.write("Protein_ID\tStrategy\tExecution_Time(s)\n")
            log_file.flush()

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
            log_file.flush()

if __name__ == '__main__':
    main()
