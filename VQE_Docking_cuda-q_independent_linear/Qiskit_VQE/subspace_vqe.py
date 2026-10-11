"""
Parallel Subspace VQE with GPU-Adaptive Execution.

This module implements a parameter-efficient VQE variant that optimizes parameters
in non-overlapping blocks, with automatic GPU detection and parallel optimization.

Key features:
- Non-overlapping block optimization with alternating offset patterns
- GPU-adaptive execution: sequential (1 GPU) or parallel (2+ GPUs)
- Alternating epoch offsets for comprehensive parameter space coverage
- ThreadPoolExecutor-based work queue for multi-GPU parallelization
- Three-level early stopping: block, global refinement, and epoch
- Optional L-BFGS-B gradient optimization with parameter-shift gradients
"""

from __future__ import annotations

from typing import List, Tuple, Callable
import csv
from contextlib import contextmanager
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cudaq
from scipy.optimize import minimize

from .efficientsu2 import efficient_su2_cudaq
from .mps_config import TARGET, configure_full_bond, configure_gpu_groups

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class SubspaceVQE:
    """
    Parallel Subspace VQE optimizer with GPU-adaptive execution.

    Instead of optimizing all parameters simultaneously, this optimizer:
    1. Divides parameters into non-overlapping blocks
    2. Uses alternating offset patterns across epochs for comprehensive coverage
    3. Automatically detects GPUs and chooses execution mode:
       - 1 GPU: Sequential block optimization
       - 2+ GPUs: Parallel block optimization with work queue
    4. Optional global refinement after each epoch
    5. Three-level early stopping for efficient convergence
    """

    def __init__(
        self,
        hamiltonian,
        num_qubits: int,
        shots: int = 8192,
        max_epochs: int = 3,
        window_size: int = 2,
        block_maxiter: int = 30,
        global_refinement: bool = True,
        global_maxiter: int = 10,
        reps: int = 4,
        entanglement: str = "circular",
        su2_gates: tuple = ("ry", "rz"),
        energy_tol: float = 1e-4,
        patience: int = 10,
        block_patience: int | None = None,
        global_patience: int | None = None,
        epoch_patience: int | None = None,
        window: int = 20,
        seed: int = 0,
        enable_parallel: bool = True,
        max_parallel_blocks: int | None = None,
        force_sequential: bool = False,
        use_hamiltonian_blocking: bool = False,
        min_block_size: int = 1,
        use_gradient_optimizer: bool = False,
        _mock_num_gpus: int | None = None,
        immediate_early_stopping: bool = False,
        timing_directory: str | None = None,
    ):
        """
        Initialize Subspace VQE optimizer.

        Args:
            hamiltonian: CUDA-Q spin operator
            num_qubits: Number of qubits in the system
            shots: Number of measurement shots (-1 for statevector)
            max_epochs: Maximum number of optimization epochs
            window_size: Number of qubits per optimization block
            block_maxiter: Max iterations per block optimization
            global_refinement: Whether to do global optimization after each epoch
            global_maxiter: Max iterations for global refinement
            reps: Ansatz repetition depth
            entanglement: Entanglement pattern for ansatz
            su2_gates: SU(2) gate sequence
            energy_tol: Energy improvement threshold for early stopping
            patience: Default patience for all levels (block, global, epoch)
            block_patience: Patience for block-level early stopping (defaults to patience)
            global_patience: Patience for global refinement early stopping (defaults to patience)
            epoch_patience: Patience for epoch-level early stopping (defaults to patience)
            window: Window size for smoothing energy improvements
            seed: Random seed for reproducibility
            enable_parallel: Enable parallel block optimization (True = auto-detect GPUs)
            max_parallel_blocks: Max number of parallel workers (None = use all GPUs)
            force_sequential: Force sequential mode even with multiple GPUs (for testing)
            use_hamiltonian_blocking: Use Hamiltonian-aware graph partitioning for blocks
            min_block_size: Minimum qubit count per block (for merging small communities)
            use_gradient_optimizer: Use L-BFGS-B with parameter-shift gradients (default: False/COBYLA)
            _mock_num_gpus: (Testing only) Override GPU count detection
            immediate_early_stopping: Stop on the first accepted energy improvement
                below energy_tol at block, global, and epoch levels. Ignores
                patience/window settings and uses an absolute energy threshold.
            timing_directory: Write evaluation and iteration CSVs here during
                optimization. Without it, timing records are retained in memory.
        """
        self.hamiltonian = hamiltonian
        self.num_qubits = int(num_qubits)
        self.shots = int(shots)
        self.max_epochs = int(max_epochs)
        self.window_size = int(window_size)
        self.block_maxiter = int(block_maxiter)
        self.global_refinement = bool(global_refinement)
        self.global_maxiter = int(global_maxiter)
        self.reps = int(reps)
        self.entanglement = entanglement
        self.su2_gates = tuple(su2_gates)
        self.energy_tol = float(energy_tol)
        self.immediate_early_stopping = bool(immediate_early_stopping)
        if self.immediate_early_stopping:
            self.patience = self.block_patience = self.global_patience = self.epoch_patience = None
            self.window = None
        else:
            # Preserve legacy smoothing/patience for callers that request it.
            self.patience = int(patience)
            self.block_patience = int(block_patience) if block_patience is not None else int(patience)
            self.global_patience = int(global_patience) if global_patience is not None else int(patience)
            self.epoch_patience = int(epoch_patience) if epoch_patience is not None else int(patience)
            self.window = int(window)
        self.seed = seed

        # Parallel execution settings
        self.enable_parallel = enable_parallel
        self.max_parallel_blocks = max_parallel_blocks
        self.force_sequential = force_sequential
        self._mock_num_gpus = _mock_num_gpus

        # Hamiltonian-aware blocking settings
        self.use_hamiltonian_blocking = use_hamiltonian_blocking
        self.min_block_size = min_block_size
        self._hamiltonian_blocks_cache = None  # Cache for graph partitioning result

        # L-BFGS-B gradient optimizer settings
        self.use_gradient_optimizer = use_gradient_optimizer

        # Detect execution mode (parallel or sequential)
        self.use_parallel, self.num_parallel_workers = self._detect_execution_mode()

        # Thread safety: Lock for GPU operations (cudaq.observe)
        # NOTE: Lock is ONLY needed for debugging/testing sequential mode
        # In mqpu mode, CUDA-Q handles concurrent observe() calls across QPUs
        # In mgpu mode, we use sequential execution (no parallelism)
        # Therefore, the lock is NOT needed for production use
        self._gpu_lock = threading.Lock() if not self.use_parallel else None

        # Metadata
        self.execution_mode = "parallel" if self.use_parallel else "sequential"
        self.num_gpus_detected = self._get_gpu_count()

        # Build ansatz
        self.kernel, self.param_order, self.param_bounds = efficient_su2_cudaq(
            num_qubits=self.num_qubits,
            reps=self.reps,
            su2_gates=self.su2_gates,
            entanglement=self.entanglement,
            skip_unentangled_qubits=False,
        )
        self.num_params = len(self.param_order)

        # Compute parameter indices per qubit
        self._compute_qubit_param_mapping()

        # Tracking
        self.energy_history: List[float] = []
        self.iteration_results: List[Tuple[float, np.ndarray]] = []
        self.block_energies: List[dict] = []
        self.epoch_summaries: List[dict] = []
        self.nfev = 0  # Number of function evaluations
        self.timing_directory = timing_directory
        self._timing_lock = threading.Lock()
        self._timing_context = threading.local()
        self._timing_start = time.perf_counter()
        self._timing_epoch = None
        self._evaluation_id = 0
        self._evaluation_seconds = 0.0
        self.energy_evaluation_timings = []
        self.iteration_timings = []
        self._timing_files = {}
        self._timing_writers = {}

    def _write_timing(self, kind, record):
        """Called under _timing_lock; stream CSVs or retain direct-solver records."""
        if self._timing_writers:
            self._timing_writers[kind].writerow(record)
            if kind == "iteration":
                for handle in self._timing_files.values():
                    handle.flush()
        elif kind == "energy_evaluation":
            self.energy_evaluation_timings.append(record)
        else:
            self.iteration_timings.append(record)

    @contextmanager
    def _timed_optimizer(self, phase, qubits=None, qpu_id=None):
        previous = getattr(self._timing_context, "optimizer", None)
        context = {
            "phase": phase, "block_qubits": ";".join(map(str, qubits or [])),
            "qpu_id": 0 if qpu_id is None else qpu_id, "epoch": self._timing_epoch,
            "last_time": time.perf_counter(), "evaluations": 0, "seconds": 0.0,
            "last_evaluations": 0, "last_seconds": 0.0, "iteration": 0,
        }
        self._timing_context.optimizer = context
        try:
            yield
        finally:
            if context["evaluations"] > context["last_evaluations"]:
                self._record_iteration(None, "optimizer_exit")
            self._timing_context.optimizer = previous

    def _record_iteration(self, energy, status="accepted"):
        context = getattr(self._timing_context, "optimizer", None)
        if context is None:
            return
        now = time.perf_counter()
        context["iteration"] += 1
        record = {
            "epoch": context["epoch"], "phase": context["phase"],
            "block_qubits": context["block_qubits"], "qpu_id": context["qpu_id"],
            "iteration": context["iteration"], "status": status,
            "start_offset_s": context["last_time"] - self._timing_start,
            "duration_s": now - context["last_time"], "energy": energy,
            "energy_evaluation_count": context["evaluations"] - context["last_evaluations"],
            "energy_evaluation_time_s": context["seconds"] - context["last_seconds"],
        }
        with self._timing_lock:
            self._write_timing("iteration", record)
        context["last_time"] = now
        context["last_evaluations"] = context["evaluations"]
        context["last_seconds"] = context["seconds"]

    def _get_gpu_count(self) -> int:
        """Get actual GPU count for metadata."""
        if self._mock_num_gpus is not None:
            return self._mock_num_gpus
        try:
            return cudaq.num_available_gpus()
        except AttributeError:
            return 1

    def _detect_execution_mode(self) -> Tuple[bool, int]:
        """
        Detect available GPUs and determine execution mode.

        Returns:
            (use_parallel, num_workers):
                - use_parallel=False → sequential with offset alternation
                - use_parallel=True → parallel with num_workers threads
        """
        # Get GPU count
        num_gpus = self._get_gpu_count()

        # Override with mock if provided
        if self._mock_num_gpus is not None:
            print(f"[TEST MODE] Mocking {num_gpus} GPUs")

        # Determine mode
        if self.force_sequential or not self.enable_parallel:
            return False, 1

        if num_gpus <= 1:
            return False, 1  # Sequential with offset alternation
        else:
            # Multi-GPU parallel mode
            if self.max_parallel_blocks is not None:
                num_workers = min(num_gpus, self.max_parallel_blocks)
            else:
                num_workers = num_gpus
            print(f"Parallel SubspaceVQE: {num_workers} GPUs detected")
            return True, num_workers

    def _compute_qubit_param_mapping(self):
        """
        Compute which parameters belong to which qubits.

        Creates a mapping: qubit_index → list of parameter indices
        This is used to extract/update parameters for specific qubits.
        """
        self.qubit_to_params: dict[int, List[int]] = {q: [] for q in range(self.num_qubits)}

        for param_idx, (param_name, qubit, layer, gate_idx) in enumerate(self.param_order):
            self.qubit_to_params[qubit].append(param_idx)

        # Verify coverage
        for q in range(self.num_qubits):
            if not self.qubit_to_params[q]:
                raise ValueError(f"No parameters found for qubit {q}. Check ansatz construction.")

    def _get_block_param_indices(self, qubits: List[int]) -> List[int]:
        """Get all parameter indices for a block of qubits."""
        indices = []
        for q in qubits:
            indices.extend(self.qubit_to_params[q])
        return sorted(indices)

    def _parse_hamiltonian_terms(self):
        """
        Parse CUDA-Q SpinOperator to extract Pauli terms and coefficients.

        Returns:
            List of (pauli_string, coefficient) tuples
            Example: [("XZIY", 0.5), ("ZIII", -0.3), ...]
        """
        terms = []

        # Try to extract terms from CUDA-Q SpinOperator
        # The exact API depends on CUDA-Q version, so we try multiple approaches
        try:
            # Newer CUDA-Q API: iterate over terms
            for term in self.hamiltonian:
                # term should have .to_string() and coefficient
                pauli_str = str(term).strip()

                # Extract coefficient (usually format: "coeff * PAULI")
                if '*' in pauli_str:
                    coeff_str, pauli_part = pauli_str.split('*', 1)
                    coeff_clean = coeff_str.strip().replace('(', '').replace(')', '')

                    # Handle complex numbers (e.g., "0.625+0i" or "0.5-0.0i")
                    # For Hermitian Hamiltonians, imaginary part should be ~0
                    if '+' in coeff_clean or 'i' in coeff_clean or 'j' in coeff_clean:
                        # Parse as complex, then take real part
                        coeff_clean = coeff_clean.replace('i', 'j')  # Python uses 'j' for imaginary
                        try:
                            coeff_complex = complex(coeff_clean)
                            coeff = coeff_complex.real
                        except ValueError:
                            # If complex parsing fails, try extracting just the real part
                            if '+' in coeff_clean:
                                real_part = coeff_clean.split('+')[0]
                            elif '-' in coeff_clean and coeff_clean.count('-') > 1:
                                # Handle negative real part like "-0.5-0i"
                                parts = coeff_clean.split('-')
                                real_part = '-' + parts[1] if parts[0] == '' else parts[0]
                            else:
                                real_part = coeff_clean.replace('j', '').replace('i', '')
                            coeff = float(real_part)
                    else:
                        coeff = float(coeff_clean)

                    pauli = pauli_part.strip()
                else:
                    # Might be just the Pauli string
                    coeff = 1.0
                    pauli = pauli_str

                terms.append((pauli, coeff))
        except (TypeError, AttributeError) as e:
            # Fallback: try to access internal representation
            # This is CUDA-Q version-dependent, may need adjustment
            try:
                # Some versions expose .to_string() method
                if hasattr(self.hamiltonian, 'to_string'):
                    ham_str = self.hamiltonian.to_string()
                    # Parse string representation (format varies)
                    # This is a simplified parser, may need refinement
                    print(f"Warning: Using fallback Hamiltonian parser")
                    print(f"Hamiltonian string: {ham_str[:200]}...")  # Show first 200 chars

                    # For now, return empty to avoid errors
                    # User should check if NetworkX is working properly
                    return []
            except Exception as e2:
                print(f"Warning: Could not parse Hamiltonian: {e2}")
                return []
        except Exception as e:
            print(f"Warning: Unexpected error parsing Hamiltonian term: {e}")
            print(f"Term string: {pauli_str}")
            # Continue with other terms

        return terms

    def _construct_interaction_graph(self):
        """
        Build weighted graph from Hamiltonian interaction structure.

        Graph structure:
        - Vertices: qubits (0 to num_qubits-1)
        - Edges: qubit pairs that appear in multi-qubit Pauli terms
        - Weights: sum of |coefficients| for terms involving the pair

        Returns:
            NetworkX Graph object
        """
        import re

        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for Hamiltonian-aware blocking. "
                "Install with: pip install networkx"
            )

        G = nx.Graph()

        # Add all qubits as nodes
        for q in range(self.num_qubits):
            G.add_node(q)

        # Parse Hamiltonian and add edges
        terms = self._parse_hamiltonian_terms()

        if not terms:
            print("Warning: No Hamiltonian terms parsed. Using geometric blocking fallback.")
            return G

        # Regex to extract qubit indices from Pauli operators
        # Matches patterns like X0, Y1, Z12, etc. (Pauli gate followed by digits)
        pauli_qubit_pattern = re.compile(r'([XYZ])(\d+)', re.IGNORECASE)

        for pauli_str, coeff in terms:
            # Find all (Pauli, qubit_index) matches in the term string
            # Example: "X0Y1Z2" → [('X', '0'), ('Y', '1'), ('Z', '2')]
            # Example: "X0 Y1" → [('X', '0'), ('Y', '1')]
            matches = pauli_qubit_pattern.findall(pauli_str)

            # Extract qubit indices (convert string digits to int)
            active_qubits = [int(qubit_str) for (pauli_char, qubit_str) in matches]

            # Add edges for all pairs in this term
            if len(active_qubits) >= 2:
                for i in range(len(active_qubits)):
                    for j in range(i + 1, len(active_qubits)):
                        qi, qj = active_qubits[i], active_qubits[j]

                        # Only add edges for valid qubit indices
                        if qi < self.num_qubits and qj < self.num_qubits:
                            if G.has_edge(qi, qj):
                                G[qi][qj]['weight'] += abs(coeff)
                            else:
                                G.add_edge(qi, qj, weight=abs(coeff))

        # Log graph statistics for debugging
        print(f"Hamiltonian graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        if G.number_of_edges() > 0:
            total_weight = sum(G[u][v]['weight'] for u, v in G.edges())
            print(f"Total edge weight: {total_weight:.4f}")

        return G

    def _partition_hamiltonian_graph(self):
        """
        Partition Hamiltonian interaction graph into blocks using community detection.

        Uses Louvain algorithm to identify weakly-interacting subsystems.

        Returns:
            List[List[int]]: Qubit blocks (same format as _generate_sliding_blocks)
        """
        if not NETWORKX_AVAILABLE:
            print("Warning: NetworkX not available. Falling back to geometric blocks.")
            return self._generate_sliding_blocks(offset=0)

        # Construct interaction graph
        G = self._construct_interaction_graph()

        # Check if graph has edges
        if G.number_of_edges() == 0:
            print("Warning: Hamiltonian graph has no edges. Using linear blocks.")
            # Fallback: create blocks of size window_size
            return self._generate_sliding_blocks(offset=0)

        # Partition using Louvain community detection
        try:
            communities = nx.community.louvain_communities(G, weight='weight', seed=self.seed)
        except AttributeError:
            # Older NetworkX version might not have louvain_communities
            print("Warning: Louvain algorithm not available. Using greedy modularity.")
            try:
                communities = nx.community.greedy_modularity_communities(G, weight='weight')
            except Exception as e:
                print(f"Warning: Graph partitioning failed ({e}). Using geometric blocks.")
                return self._generate_sliding_blocks(offset=0)

        # Convert communities to sorted qubit lists
        blocks = [sorted(list(comm)) for comm in communities]

        # Merge blocks that are too small
        blocks = self._merge_small_blocks(blocks)

        # Sort blocks by first qubit index for consistent ordering
        blocks.sort(key=lambda b: b[0])

        return blocks

    def _merge_small_blocks(self, blocks: List[List[int]]) -> List[List[int]]:
        """
        Merge blocks smaller than min_block_size with neighboring blocks.

        Args:
            blocks: List of qubit blocks

        Returns:
            Merged blocks
        """
        if not blocks:
            return blocks

        # Sort by first qubit
        blocks = sorted(blocks, key=lambda b: b[0] if b else 0)

        merged = []
        current_block = blocks[0].copy()

        for i in range(1, len(blocks)):
            if len(current_block) < self.min_block_size:
                # Merge with next block
                current_block.extend(blocks[i])
                current_block = sorted(current_block)
            else:
                # Current block is large enough, save it
                merged.append(current_block)
                current_block = blocks[i].copy()

        # Don't forget the last block
        merged.append(current_block)

        return merged

    def _evaluate_energy(self, params: np.ndarray, qpu_id: int | None = None, use_hamiltonian_batching: bool = False,
                         evaluation_kind="energy", parameter_index=None, parameter_shift=None,
                         kernel=None, hamiltonian=None) -> float:
        """
        Evaluate energy for given parameter vector.

        Thread-safety:
        - mqpu mode (parallel): Uses qpu_id for explicit GPU assignment (no lock)
        - mgpu/sequential mode: Lock protects sequential execution (if enabled)

        Args:
            params: Parameter vector
            qpu_id: GPU ID for mqpu mode (None for sequential/automatic)
            use_hamiltonian_batching: If True and multiple GPUs available, distribute
                Hamiltonian terms across GPUs for parallel evaluation (used in global refinement)
        """
        kernel = self.kernel if kernel is None else kernel
        hamiltonian = self.hamiltonian if hamiltonian is None else hamiltonian
        call_start = time.perf_counter()
        with self._timing_lock:
            self._evaluation_id += 1
            evaluation_id = self._evaluation_id
        params_list = params.flatten().tolist()

        # Use lock only in sequential mode (mgpu or force_sequential)
        # In mqpu parallel mode, each thread uses explicit qpu_id
        lock_start = time.perf_counter()
        if self._gpu_lock is not None:
            self._gpu_lock.acquire()
        evaluation_start = time.perf_counter()
        energy = None
        status, error = "error", ""

        try:
            # Three execution modes:
            # 1. Hamiltonian batching: Distribute Hamiltonian terms across GPUs (global refinement)
            # 2. Block parallelization: Assign specific block to specific GPU (block optimization)
            # 3. Sequential: Standard single-GPU execution (fallback)

            if cudaq.get_target().name == TARGET:
                # The custom backend shards the fixed shot budget itself.
                # CUDA-Q's parallel.thread would split terms a second time.
                kwargs = {"qpu_id": 0 if qpu_id is None else int(qpu_id)}
                if self.shots != -1:
                    kwargs["shots_count"] = self.shots
                obs_result = cudaq.observe_async(
                    kernel, hamiltonian, *params_list, **kwargs
                ).get()
            elif use_hamiltonian_batching and self.num_gpus_detected > 1:
                # Mode 1: Hamiltonian batching for global refinement
                # Distribute Hamiltonian terms across all available GPUs
                # No qpu_id - CUDA-Q automatically distributes terms
                try:
                    if self.shots == -1:
                        obs_result = cudaq.observe(
                            kernel, hamiltonian, *params_list,
                            execution=cudaq.parallel.thread
                        )
                    else:
                        obs_result = cudaq.observe(
                            kernel, hamiltonian, *params_list,
                            shots_count=self.shots,
                            execution=cudaq.parallel.thread
                        )
                except (TypeError, AttributeError):
                    # Fallback if execution parameter not supported
                    if self.shots == -1:
                        obs_result = cudaq.observe(kernel, hamiltonian, *params_list)
                    else:
                        obs_result = cudaq.observe(
                            kernel, hamiltonian, *params_list,
                            shots_count=self.shots
                        )

            elif qpu_id is not None:
                # Mode 2: Block parallelization
                # Assign this block optimization to a specific GPU
                try:
                    if self.shots == -1:
                        obs_result = cudaq.observe(
                            kernel, hamiltonian, *params_list,
                            execution=cudaq.parallel.thread, qpu_id=qpu_id
                        )
                    else:
                        obs_result = cudaq.observe(
                            kernel, hamiltonian, *params_list,
                            shots_count=self.shots,
                            execution=cudaq.parallel.thread, qpu_id=qpu_id
                        )
                except TypeError:
                    # qpu_id not supported, try execution parameter only
                    # CUDA-Q should automatically distribute across available GPUs
                    try:
                        if self.shots == -1:
                            obs_result = cudaq.observe(
                                kernel, hamiltonian, *params_list,
                                execution=cudaq.parallel.thread
                            )
                        else:
                            obs_result = cudaq.observe(
                                kernel, hamiltonian, *params_list,
                                shots_count=self.shots,
                                execution=cudaq.parallel.thread
                            )
                    except (TypeError, AttributeError):
                        # execution parameter not supported either
                        # Fall back to standard observe (mqpu backend handles distribution)
                        if self.shots == -1:
                            obs_result = cudaq.observe(kernel, hamiltonian, *params_list)
                        else:
                            obs_result = cudaq.observe(
                                kernel, hamiltonian, *params_list, shots_count=self.shots
                            )
            else:
                # Mode 3: Sequential execution
                # Standard single-GPU observe call
                if self.shots == -1:
                    obs_result = cudaq.observe(kernel, hamiltonian, *params_list)
                else:
                    obs_result = cudaq.observe(
                        kernel, hamiltonian, *params_list, shots_count=self.shots
                    )

            # Extract energy
            if hasattr(obs_result, "expectation_value"):
                energy = float(obs_result.expectation_value)
            elif hasattr(obs_result, "expectation"):
                exp_attr = getattr(obs_result, "expectation")
                energy = float(exp_attr() if callable(exp_attr) else exp_attr)
            elif hasattr(obs_result, "energy"):
                energy = float(obs_result.energy)
            else:
                energy = float(obs_result)

            status = "ok"
            return energy
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            finished = time.perf_counter()
            if self._gpu_lock is not None:
                self._gpu_lock.release()
            context = getattr(self._timing_context, "optimizer", None)
            duration = finished - evaluation_start
            if context is not None:
                context["evaluations"] += 1
                context["seconds"] += duration
            record = {
                "evaluation_id": evaluation_id, "epoch": self._timing_epoch,
                "phase": context["phase"] if context else ("initial" if self._timing_epoch is None else "fused"),
                "block_qubits": context["block_qubits"] if context else "",
                "qpu_id": 0 if qpu_id is None else qpu_id,
                "kind": evaluation_kind, "parameter_index": parameter_index,
                "parameter_shift": parameter_shift, "shots": self.shots,
                "start_offset_s": call_start - self._timing_start,
                "lock_wait_s": evaluation_start - lock_start,
                "duration_s": duration, "total_s": finished - call_start,
                "energy": energy, "status": status, "error": error,
            }
            with self._timing_lock:
                self._evaluation_seconds += duration
                if status == "ok":
                    self.nfev += 1
                self._write_timing("energy_evaluation", record)

    class EarlyStop(Exception):
        """Internal sentinel exception for early stopping."""

    def _compute_block_gradient(
        self,
        theta_full: np.ndarray,
        block_param_indices: List[int],
        qpu_id: int | None = None,
    ) -> np.ndarray:
        """
        Compute gradient for block parameters using parameter-shift rule.

        Only computes gradients for the parameters in the block, keeping others fixed.

        Args:
            theta_full: Full parameter vector
            block_param_indices: Indices of parameters to compute gradients for
            qpu_id: GPU ID for mqpu mode (None for sequential/automatic)

        Returns:
            gradient: Gradient vector for block parameters only
        """
        n_block_params = len(block_param_indices)
        gradient = np.zeros(n_block_params, dtype=float)
        shift = np.pi / 2

        for i, param_idx in enumerate(block_param_indices):
            # Forward shift
            params_plus = theta_full.copy()
            params_plus[param_idx] += shift
            energy_plus = self._evaluate_energy(params_plus, qpu_id=qpu_id,
                evaluation_kind="gradient", parameter_index=param_idx, parameter_shift=shift)

            # Backward shift
            params_minus = theta_full.copy()
            params_minus[param_idx] -= shift
            energy_minus = self._evaluate_energy(params_minus, qpu_id=qpu_id,
                evaluation_kind="gradient", parameter_index=param_idx, parameter_shift=-shift)

            # Parameter-shift gradient
            gradient[i] = (energy_plus - energy_minus) / 2.0

        return gradient

    def _optimize_block(
        self, theta_full: np.ndarray, qubits: List[int], maxiter: int,
        qpu_id: int | None = None,
    ) -> Tuple[np.ndarray, float]:
        with self._timed_optimizer("block", qubits, qpu_id):
            return self._optimize_block_impl(theta_full, qubits, maxiter, qpu_id)

    def _optimize_block_impl(
        self,
        theta_full: np.ndarray,
        qubits: List[int],
        maxiter: int,
        qpu_id: int | None = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize parameters for a specific block of qubits with early stopping.

        Supports two optimization methods:
        - COBYLA (default): Derivative-free optimization
        - L-BFGS-B (use_gradient_optimizer=True): Gradient-based with parameter-shift rule

        Args:
            theta_full: Full parameter vector (all qubits)
            qubits: List of qubit indices for this block
            maxiter: Max iterations for this block
            qpu_id: GPU ID for parallel execution (None for sequential)

        Returns:
            theta_optimized: Updated full parameter vector with best parameters
            energy: Best (lowest) energy after optimization
        """
        # Get parameter indices for this block
        block_param_indices = self._get_block_param_indices(qubits)

        # Extract block parameters
        theta_block = theta_full[block_param_indices]

        # Early stopping state and parameter tracking
        block_energy_history = []
        block_params_history = []
        no_improve_count = 0
        previous_accepted_energy = None

        # Cost function: only optimize block parameters, fix others
        def cost_func_block(theta_b):
            theta_temp = theta_full.copy()
            theta_temp[block_param_indices] = theta_b
            # Pass qpu_id for parallel execution
            energy = self._evaluate_energy(theta_temp, qpu_id=qpu_id, evaluation_kind="objective")
            block_energy_history.append(energy)
            block_params_history.append(theta_b.copy())
            return energy

        # Gradient function for L-BFGS-B: compute gradients for block parameters only
        def gradient_func_block(theta_b):
            theta_temp = theta_full.copy()
            theta_temp[block_param_indices] = theta_b
            gradient = self._compute_block_gradient(theta_temp, block_param_indices, qpu_id=qpu_id)
            return gradient

        # Early stopping callback
        def callback(_xk):
            nonlocal no_improve_count, previous_accepted_energy
            n = len(block_energy_history)
            if n:
                accepted_energy = next(
                    (energy for energy, parameters in zip(
                        reversed(block_energy_history), reversed(block_params_history)
                    ) if np.array_equal(parameters, _xk)), block_energy_history[-1]
                )
                self._record_iteration(accepted_energy)
            if self.immediate_early_stopping:
                if not block_energy_history:
                    return
                if previous_accepted_energy is None:
                    previous_accepted_energy = block_energy_history[0]
                # A line search can evaluate a rejected trial after the point
                # passed to this callback. Compare accepted iterates only.
                current = next(
                    energy for energy, parameters in zip(
                        reversed(block_energy_history), reversed(block_params_history)
                    ) if np.array_equal(parameters, _xk)
                )
                improvement = previous_accepted_energy - current
                previous_accepted_energy = current
                if improvement < self.energy_tol:
                    raise SubspaceVQE.EarlyStop()
                return
            if n < 2:
                return

            # Check recent window of energies
            window_size = min(self.window, n - 1)
            recent = block_energy_history[-(window_size + 1):]
            diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
            best_recent_improve = max(diffs) if diffs else float("inf")

            # Energy convergence check
            converged = best_recent_improve < self.energy_tol

            if converged:
                no_improve_count += 1
            else:
                no_improve_count = 0

            # Stop if patience exhausted
            if no_improve_count >= self.block_patience:
                raise SubspaceVQE.EarlyStop()

        # Choose optimization method
        if self.use_gradient_optimizer:
            # L-BFGS-B with parameter-shift gradients
            # Derive tolerances from energy_tol (consistent with CUDAQVQE)
            optimizer_method = "L-BFGS-B"
            optimizer_options = {
                "maxiter": maxiter,
                "ftol": 1e-12 if self.immediate_early_stopping else self.energy_tol,
                "gtol": 1e-8 if self.immediate_early_stopping else self.energy_tol * 0.1,
                "disp": False,
            }
            jac = gradient_func_block
            # Parameter bounds for L-BFGS-B (unbounded by default)
            bounds = None
        else:
            # COBYLA (derivative-free)
            optimizer_method = "COBYLA"
            optimizer_options = {"maxiter": maxiter, "disp": False}
            jac = None
            bounds = None

        # Optimize with early stopping
        try:
            result = minimize(
                cost_func_block,
                theta_block,
                method=optimizer_method,
                jac=jac,
                bounds=bounds,
                options=optimizer_options,
                callback=callback,
            )
            theta_optimized = theta_full.copy()
            theta_optimized[block_param_indices] = result.x
            final_energy = float(result.fun)
        except SubspaceVQE.EarlyStop:
            # Early stop triggered - use best result from history
            best_idx = int(np.argmin(block_energy_history))
            best_block_params = block_params_history[best_idx]
            theta_optimized = theta_full.copy()
            theta_optimized[block_param_indices] = best_block_params
            final_energy = block_energy_history[best_idx]
            method_name = "L-BFGS-B" if self.use_gradient_optimizer else "COBYLA"
            print(f"      Early stop ({method_name}) at {len(block_energy_history)} iters (ΔE < {self.energy_tol:.2e})")

        # Ensure we return the best parameters seen during optimization
        if block_energy_history:
            best_idx = int(np.argmin(block_energy_history))
            if block_energy_history[best_idx] < final_energy:
                best_block_params = block_params_history[best_idx]
                theta_optimized = theta_full.copy()
                theta_optimized[block_param_indices] = best_block_params
                final_energy = block_energy_history[best_idx]

        return theta_optimized, final_energy

    def _parallel_block_optimization(
        self,
        theta_init: np.ndarray,
        blocks: List[List[int]],
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize multiple blocks in parallel using ThreadPoolExecutor.

        Uses round-robin QPU assignment to distribute work across GPUs.

        Args:
            theta_init: Starting parameter vector
            blocks: List of qubit blocks

        Returns:
            theta_final: Updated parameters
            energies: Energy after each block
        """
        num_blocks = len(blocks)
        num_workers = min(self.num_parallel_workers, num_blocks)
        custom_mps = cudaq.get_target().name == TARGET
        if custom_mps:
            configure_gpu_groups(self.num_gpus_detected, num_workers)

        print(f"  Parallel scan: {num_blocks} blocks on {num_workers} GPUs")

        block_results = {}
        results_lock = threading.Lock()
        qpu_locks = [threading.Lock() for _ in range(num_workers)]

        def optimize_block_worker(block_idx: int, block_qubits: List[int]):
            """Worker: optimize one block on assigned GPU."""
            # Assign QPU ID using round-robin across available GPUs
            qpu_id = block_idx % num_workers

            # Optimize block with explicit QPU assignment
            # This ensures CUDA-Q uses the specific GPU
            # Note: theta_init is read-only, no lock needed
            # A queued block may share its modulo QPU with a slower block.
            # Reserve that QPU for the entire block optimization.
            with qpu_locks[qpu_id]:
                theta_opt, energy = self._optimize_block(
                    theta_init,
                    block_qubits,
                    self.block_maxiter,
                    qpu_id=qpu_id,
                )

            return block_idx, theta_opt, energy

        # Wait for every worker before restoring the sequential GPU group,
        # including when a block raises an optimizer/backend error.
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(optimize_block_worker, idx, block): idx
                    for idx, block in enumerate(blocks)
                }

                for future in as_completed(futures):
                    block_idx, theta_opt, energy = future.result()
                    qpu_id = block_idx % num_workers
                    with results_lock:
                        block_results[block_idx] = (theta_opt, energy)
                    print(f"    Block {block_idx+1}/{num_blocks} {blocks[block_idx]} [GPU {qpu_id}]: E = {energy:.6f}")
        finally:
            if custom_mps:
                configure_gpu_groups(self.num_gpus_detected)

        # Reconstruct sequential updates (apply block updates in order)
        theta_current = theta_init.copy()
        energies = []

        for i in range(num_blocks):
            theta_opt, energy = block_results[i]
            block_param_indices = self._get_block_param_indices(blocks[i])
            theta_current[block_param_indices] = theta_opt[block_param_indices]
            energies.append(energy)
            self.energy_history.append(energy)

        return theta_current, energies

    def _generate_sliding_blocks(self, offset: int = 0) -> List[List[int]]:
        """
        Generate non-overlapping blocks with configurable offset.

        Args:
            offset: Starting position offset (0 or 1 for alternation)

        Examples:
            7 qubits, window_size=2:
            offset=0: [[0,1], [2,3], [4,5], [6]]       - non-overlapping blocks
            offset=1: [[0], [1,2], [3,4], [5,6]]      - single qubit first, then non-overlapping
        """
        blocks = []

        if offset == 0:
            # Standard non-overlapping blocks
            for start in range(0, self.num_qubits, self.window_size):
                end = min(start + self.window_size, self.num_qubits)
                block_qubits = list(range(start, end))
                blocks.append(block_qubits)
        else:
            # Offset=1: Start with single qubit [0], then window_size blocks from position 1
            if self.num_qubits > 0:
                blocks.append([0])  # First block is just qubit 0
                # Continue with window_size blocks starting from position 1
                for start in range(1, self.num_qubits, self.window_size):
                    end = min(start + self.window_size, self.num_qubits)
                    block_qubits = list(range(start, end))
                    blocks.append(block_qubits)

        return blocks

    def _sequential_block_optimization(
        self, theta_init: np.ndarray, blocks: List[List[int]]
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize blocks sequentially (one at a time).

        All blocks start from the same theta_init (no cumulative updates during scan).
        Updates are applied sequentially after all blocks are optimized.

        Args:
            theta_init: Starting parameter vector
            blocks: List of qubit blocks to optimize

        Returns:
            theta_optimized: Parameter vector after all block optimizations
            energies: Energy after each block optimization
        """
        block_results = []

        print(f"  Sequential scan: {len(blocks)} blocks")

        # Optimize each block from the same starting point
        for i, block_qubits in enumerate(blocks):
            theta_opt, energy = self._optimize_block(theta_init, block_qubits, self.block_maxiter)
            block_results.append((theta_opt, energy))
            print(f"    Block {i+1}/{len(blocks)} {block_qubits}: E = {energy:.6f}")

        # Reconstruct sequential updates (apply block updates in order)
        theta_current = theta_init.copy()
        energies = []

        for i, (theta_opt, energy) in enumerate(block_results):
            block_param_indices = self._get_block_param_indices(blocks[i])
            theta_current[block_param_indices] = theta_opt[block_param_indices]
            energies.append(energy)
            self.energy_history.append(energy)

        return theta_current, energies

    def _forward_scan(self, theta_init: np.ndarray, offset: int = 0) -> Tuple[np.ndarray, List[float]]:
        """
        Forward sliding window scan: optimize blocks from left to right.

        This is a convenience wrapper around _sequential_block_optimization
        that generates geometric blocks with the specified offset.

        Args:
            theta_init: Starting parameter vector
            offset: Offset for block generation (0 or 1)

        Returns:
            theta_forward: Parameter vector after forward scan
            energies: Energy after each block optimization
        """
        blocks = self._generate_sliding_blocks(offset=offset)
        return self._sequential_block_optimization(theta_init, blocks)


    def _compute_full_gradient(
        self,
        theta: np.ndarray,
        use_hamiltonian_batching: bool = False,
    ) -> np.ndarray:
        """
        Compute gradient for all parameters using parameter-shift rule.

        Args:
            theta: Full parameter vector
            use_hamiltonian_batching: Distribute Hamiltonian terms across GPUs

        Returns:
            gradient: Full gradient vector
        """
        n_params = len(theta)
        gradient = np.zeros(n_params, dtype=float)
        shift = np.pi / 2

        for i in range(n_params):
            # Forward shift
            params_plus = theta.copy()
            params_plus[i] += shift
            energy_plus = self._evaluate_energy(params_plus, use_hamiltonian_batching=use_hamiltonian_batching,
                evaluation_kind="gradient", parameter_index=i, parameter_shift=shift)

            # Backward shift
            params_minus = theta.copy()
            params_minus[i] -= shift
            energy_minus = self._evaluate_energy(params_minus, use_hamiltonian_batching=use_hamiltonian_batching,
                evaluation_kind="gradient", parameter_index=i, parameter_shift=-shift)

            # Parameter-shift gradient
            gradient[i] = (energy_plus - energy_minus) / 2.0

        return gradient

    def _global_refinement(self, theta_init: np.ndarray) -> Tuple[np.ndarray, float]:
        with self._timed_optimizer("global"):
            return self._global_refinement_impl(theta_init)

    def _global_refinement_impl(self, theta_init: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Global optimization on all parameters (short refinement with early stopping).

        Uses Hamiltonian batching to distribute Hamiltonian terms across multiple GPUs
        for accelerated energy evaluation.

        Supports two optimization methods:
        - COBYLA (default): Derivative-free optimization
        - L-BFGS-B (use_gradient_optimizer=True): Gradient-based with parameter-shift rule

        Args:
            theta_init: Starting parameters

        Returns:
            theta_refined: Best parameters found during refinement
            energy: Best (lowest) energy
        """
        method_name = "L-BFGS-B" if self.use_gradient_optimizer else "COBYLA"
        print(f"  ⚙️  Global refinement ({method_name}, maxiter={self.global_maxiter})")

        # Determine if we should use Hamiltonian batching
        # Only enable if:
        # 1. Multiple GPUs are available
        # 2. Not in forced sequential mode
        # 3. Parallel execution is enabled
        use_batching = (
            self.num_gpus_detected > 1
            and not self.force_sequential
            and self.use_parallel
        )

        if use_batching:
            print(f"    Hamiltonian batching: distributing terms across {self.num_gpus_detected} GPUs")
        else:
            print(f"    Sequential mode: using single GPU")

        # Early stopping state for global refinement
        global_energy_history = []
        global_params_history = []
        global_no_improve_count = 0
        previous_accepted_energy = None

        def cost_func_global(xk):
            # Enable Hamiltonian batching for multi-GPU acceleration
            en = self._evaluate_energy(xk, use_hamiltonian_batching=use_batching,
                                       evaluation_kind="objective")
            global_energy_history.append(en)
            global_params_history.append(np.array(xk, dtype=float))
            return en

        def gradient_func_global(xk):
            """Compute full gradient using parameter-shift rule."""
            return self._compute_full_gradient(xk, use_hamiltonian_batching=use_batching)

        def callback_global(_xk):
            nonlocal global_no_improve_count, previous_accepted_energy
            n = len(global_energy_history)
            if n:
                accepted_energy = next(
                    (energy for energy, parameters in zip(
                        reversed(global_energy_history), reversed(global_params_history)
                    ) if np.array_equal(parameters, _xk)), global_energy_history[-1]
                )
                self._record_iteration(accepted_energy)
            if self.immediate_early_stopping:
                if not global_energy_history:
                    return
                if previous_accepted_energy is None:
                    previous_accepted_energy = global_energy_history[0]
                current = next(
                    energy for energy, parameters in zip(
                        reversed(global_energy_history), reversed(global_params_history)
                    ) if np.array_equal(parameters, _xk)
                )
                improvement = previous_accepted_energy - current
                previous_accepted_energy = current
                if improvement < self.energy_tol:
                    raise SubspaceVQE.EarlyStop()
                return
            if n > self.window:
                # Check if best improvement in recent window < energy_tol
                window_size = min(self.window, n - 1)
                recent = global_energy_history[-(window_size + 1):]
                diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
                best_recent_improve = max(diffs) if diffs else float("inf")

                if best_recent_improve < self.energy_tol:
                    global_no_improve_count += 1
                else:
                    global_no_improve_count = 0

                if global_no_improve_count >= self.global_patience:
                    print(f"    ✓ Global refinement early stop ({method_name}, iter {n})")
                    raise SubspaceVQE.EarlyStop()

        # Choose optimization method
        if self.use_gradient_optimizer:
            # L-BFGS-B with parameter-shift gradients
            # Derive tolerances from energy_tol (consistent with CUDAQVQE)
            optimizer_method = "L-BFGS-B"
            optimizer_options = {
                "maxiter": self.global_maxiter,
                "ftol": 1e-12 if self.immediate_early_stopping else self.energy_tol,
                "gtol": 1e-8 if self.immediate_early_stopping else self.energy_tol * 0.1,
                "disp": False,
            }
            jac = gradient_func_global
            bounds = None
        else:
            # COBYLA (derivative-free)
            optimizer_method = "COBYLA"
            optimizer_options = {"maxiter": self.global_maxiter, "disp": False}
            jac = None
            bounds = None

        try:
            result = minimize(
                cost_func_global,
                theta_init,
                method=optimizer_method,
                jac=jac,
                bounds=bounds,
                options=optimizer_options,
                callback=callback_global,
            )
            theta_refined = np.array(result.x, dtype=float)
            energy = float(result.fun)
        except SubspaceVQE.EarlyStop:
            # Use best result from history
            best_idx = int(np.argmin(global_energy_history))
            theta_refined = global_params_history[best_idx]
            energy = global_energy_history[best_idx]

        # Ensure we return the best parameters seen during refinement
        if global_energy_history:
            best_idx = int(np.argmin(global_energy_history))
            if global_energy_history[best_idx] < energy:
                theta_refined = global_params_history[best_idx]
                energy = global_energy_history[best_idx]

        self.energy_history.append(energy)
        print(f"    Refined energy: {energy:.6f}")

        return theta_refined, energy

    def run_vqe(self, init_params: np.ndarray | None = None):
        """Run optimization, recording wall-clock timings with perf_counter."""
        self._timing_start = time.perf_counter()
        self._timing_epoch = None
        self._evaluation_id = 0
        self._evaluation_seconds = 0.0
        self.energy_evaluation_timings = []
        self.iteration_timings = []
        if self.timing_directory is not None:
            directory = Path(self.timing_directory)
            directory.mkdir(parents=True, exist_ok=True)
        try:
            if self.timing_directory is not None:
                fields = {
                    "energy_evaluation": ["evaluation_id", "epoch", "phase", "block_qubits", "qpu_id",
                        "kind", "parameter_index", "parameter_shift", "shots", "start_offset_s",
                        "lock_wait_s", "duration_s", "total_s", "energy", "status", "error"],
                    "iteration": ["epoch", "phase", "block_qubits", "qpu_id", "iteration", "status",
                        "start_offset_s", "duration_s", "energy", "energy_evaluation_count",
                        "energy_evaluation_time_s"],
                }
                for kind, columns in fields.items():
                    handle = (directory / f"{kind}_timings.csv").open("w", newline="")
                    self._timing_files[kind] = handle
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    self._timing_writers[kind] = writer
            return self._run_vqe(init_params)
        finally:
            for handle in self._timing_files.values():
                handle.close()
            self._timing_files = {}
            self._timing_writers = {}

    def _run_vqe(self, init_params: np.ndarray | None = None):
        """
        Run Parallel SubspaceVQE optimization with GPU-adaptive execution.

        Algorithm:
            For each epoch:
                1. Generate blocks with alternating offset (0 or 1)
                2. Sequential mode (1 GPU): Optimize blocks sequentially
                   Parallel mode (2+ GPUs): Optimize blocks in parallel
                3. Global refinement (optional)
                4. Check convergence

        Args:
            init_params: Initial parameter guess (random if None)

        Returns:
            energy_list: All energies observed during optimization
            best_params: Best parameter vector found
            kernel: CUDA-Q ansatz kernel
            top_results: Top 6 (energy, params) pairs
        """
        # Initialize
        if cudaq.get_target().name == TARGET:
            configure_full_bond(self.num_qubits, [
                {"entanglement": self.entanglement, "reps": self.reps}
            ])
        np.random.seed(self.seed)
        if init_params is None:
            theta = np.random.uniform(-np.pi, np.pi, self.num_params)
        else:
            theta = np.array(init_params, dtype=float)

        self.energy_history = []
        self.iteration_results = []
        self.block_energies = []
        self.epoch_summaries = []
        self.nfev = 0

        # Initial energy
        energy_init = self._evaluate_energy(theta)
        self.energy_history.append(energy_init)
        self.iteration_results.append((energy_init, theta.copy()))
        print(f"\n{'='*60}")
        print(f"Subspace VQE: Block-wise Optimization")
        print(f"{'='*60}")
        print(f"Execution mode: {self.execution_mode}")
        if self.use_parallel:
            print(f"Parallel workers: {self.num_parallel_workers}")
        optimizer_name = "L-BFGS-B (gradient)" if self.use_gradient_optimizer else "COBYLA (derivative-free)"
        print(f"Optimizer: {optimizer_name}")
        print(f"Parameters: {self.num_params}")

        # Display blocking strategy
        if self.use_hamiltonian_blocking:
            print(f"Blocking strategy: Hamiltonian-aware (graph partitioning)")
            # Cache the Hamiltonian blocks for efficiency
            if self._hamiltonian_blocks_cache is None:
                self._hamiltonian_blocks_cache = self._partition_hamiltonian_graph()
            print(f"Blocks: {self._hamiltonian_blocks_cache}")
            print(f"Number of blocks: {len(self._hamiltonian_blocks_cache)}")
        else:
            print(f"Blocking strategy: Geometric (window_size={self.window_size})")
            print(f"Blocks per scan: {len(self._generate_sliding_blocks())}")

        print(f"Initial energy: {energy_init:.6f}")
        print(f"{'='*60}\n")

        best_energy = energy_init
        best_params = theta.copy()

        # Early stopping state for epochs
        epoch_energy_history = [energy_init]
        epoch_no_improve_count = 0
        previous_epoch_energy = energy_init

        # Main optimization loop
        for epoch in range(self.max_epochs):
            self._timing_epoch = epoch
            epoch_start = time.perf_counter()
            epoch_evaluations = self._evaluation_id
            epoch_evaluation_seconds = self._evaluation_seconds
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            print(f"{'='*60}")

            # Generate blocks based on blocking strategy
            if self.use_hamiltonian_blocking:
                # Hamiltonian-aware: use cached graph partitioning
                blocks = self._hamiltonian_blocks_cache
                epoch_offset = None  # Not applicable for Hamiltonian-aware
                print(f"Hamiltonian-aware blocks: {blocks}")
            else:
                # Geometric: use alternating offset pattern
                epoch_offset = epoch % 2
                blocks = self._generate_sliding_blocks(offset=epoch_offset)
                print(f"Geometric blocks (offset={epoch_offset}): {blocks}")

            # Optimize blocks (parallel or sequential)
            if not self.use_parallel:
                # Sequential mode: optimize blocks one by one
                print(f"Sequential mode")
                theta_fused, energies = self._sequential_block_optimization(theta, blocks)

            else:
                # Parallel mode: optimize blocks in parallel
                print(f"Parallel mode")
                theta_fused, energies = self._parallel_block_optimization(theta, blocks)

            # Individual block energies belong to different full parameter
            # vectors. Their simultaneous updates need a fresh full-H evaluation.
            energy_current = self._evaluate_energy(theta_fused)
            self.energy_history.append(energy_current)

            # Update current parameters
            theta = theta_fused

            # Global refinement
            if self.global_refinement:
                theta, energy_refined = self._global_refinement(theta)
            else:
                energy_refined = energy_current

            # Track best and check for improvement (for epoch-level early stopping)
            found_new_best = False
            if energy_refined < best_energy:
                best_energy = energy_refined
                best_params = theta.copy()
                found_new_best = True
                print(f"  ✓ New best energy: {best_energy:.6f}")

            # Store iteration result
            self.iteration_results.append((energy_refined, theta.copy()))

            # Reset to best params for next epoch (always start from historical best)
            theta = best_params.copy()

            # Epoch summary
            epoch_summary = {
                "epoch": epoch,
                "offset": epoch_offset,
                "energy_current": energy_current,
                "energy_refined": energy_refined,
                "best_energy": best_energy,
                "duration_s": time.perf_counter() - epoch_start,
                "energy_evaluation_count": self._evaluation_id - epoch_evaluations,
                "energy_evaluation_time_s": self._evaluation_seconds - epoch_evaluation_seconds,
            }

            self.epoch_summaries.append(epoch_summary)

            print(f"\n  Epoch summary:")
            print(f"    Current: {energy_current:.6f}")
            print(f"    Refined: {energy_refined:.6f}")
            print(f"    Best so far: {best_energy:.6f}")

            # Epoch-level early stopping: check if we found a new historical best
            if self.immediate_early_stopping:
                improvement = previous_epoch_energy - energy_refined
                previous_epoch_energy = energy_refined
                if improvement < self.energy_tol:
                    print(f"\n  ✓ Epoch early stopping: energy improvement {improvement:.6f} < {self.energy_tol}")
                    break
                continue
            if epoch >= 0:  # Start checking from epoch 0
                if found_new_best:
                    # Reset counter when we find a new best
                    epoch_no_improve_count = 0
                else:
                    # Increment counter when no improvement
                    epoch_no_improve_count += 1
                    print(f"  ⚠️  No improvement for {epoch_no_improve_count} consecutive epoch(s)")

                # Stop if patience exhausted
                if epoch_no_improve_count >= self.epoch_patience:
                    print(f"\n  ✓ Epoch early stopping: No new best energy for {self.epoch_patience} consecutive epochs")
                    break

        # Final summary - ensure we have the absolute best from all iterations
        if self.iteration_results:
            sorted_results = sorted(self.iteration_results, key=lambda x: x[0])
            absolute_best_energy, absolute_best_params = sorted_results[0]
            if absolute_best_energy < best_energy:
                best_energy = absolute_best_energy
                best_params = absolute_best_params

        print(f"\n{'='*60}")
        print(f"Optimization Complete")
        print(f"{'='*60}")
        print(f"Total function evaluations: {self.nfev}")
        print(f"Total epochs: {len(self.epoch_summaries)}")
        print(f"Best energy: {best_energy:.6f}")
        print(f"Initial energy: {energy_init:.6f}")
        print(f"Total improvement: {energy_init - best_energy:.6f}")
        print(f"{'='*60}\n")

        # Top results
        top_results = sorted(self.iteration_results, key=lambda x: x[0])[:6]

        return self.energy_history, best_params, self.kernel, top_results


__all__ = ["SubspaceVQE"]
