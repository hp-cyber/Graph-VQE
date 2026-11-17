from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, is_dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import cudaq
import numpy as np
from cudaq import spin
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from .custom_ansatz import improved_efficient_su2_cudaq
from .efficientsu2 import efficient_su2_cudaq
from .utils_scaling import CompoundScaleConfig, compound_scale
from .gradient_optimizer import GradientBasedVQEOptimizer

# Type aliases for clarity
EntanglementPattern = Union[str, List[Tuple[int, int]], Callable[[int, int], List[Tuple[int, int]]]]


def qiskit_sparsepauliop_to_cudaq_spin(op: SparsePauliOp, reverse_qubit_order: bool = True):
    """
    Convert Qiskit's SparsePauliOp to a CUDA-Q SpinOperator.
    reverse_qubit_order=True maps Qiskit's rightmost qubit (q0) to CUDA-Q index 0.
    """
    n = op.num_qubits
    H = 0.0
    for label, coeff in op.to_list():  # e.g. 'ZIXY'
        if coeff == 0:
            continue
        term = 1.0
        for idx, ch in enumerate(label):
            q = (n - 1 - idx) if reverse_qubit_order else idx
            if ch == "I":
                continue
            elif ch == "X":
                term = term * spin.x(q)
            elif ch == "Y":
                term = term * spin.y(q)
            elif ch == "Z":
                term = term * spin.z(q)
        H = H + (float(coeff.real) * term)
    return H


class CUDAQVQE:
    """
    CUDA-Q based VQE solver with optional EfficientNet-style compound scaling.

    Key features:
      * Flexible ansatz selection (standard or alternating-CX variant)
      * Early stopping via energy tolerance, patience, and recent window smoothing
      * Optional staged training schedule driven by compound scaling on depth / entanglement / shots
      * Stage-level metadata for downstream evaluation (energy trajectories, configurations, etc.)
      * Optional Hamiltonian batching for multi-GPU acceleration
    """

    class EarlyStop(Exception):
        """Internal sentinel exception used to abort SciPy optimization early."""

    def __init__(
        self,
        hamiltonian,
        optimization_level: int = 3,
        shots: int = 200,
        max_iterations: int = 20,
        seed: int | None = None,
        energy_tol: float | None = None,
        patience: int = 5,
        window: int = 1,
        *,
        reps: int | None = None,
        entanglement: EntanglementPattern | None = None,
        skip_unentangled_qubits: bool | None = None,
        su2_gates: Sequence[str | Callable] = ("ry", "rz"),
        ansatz_variant: str = "efficientsu2",
        phi_schedule: Sequence[int] | None = None,
        scale_config: CompoundScaleConfig | None = None,
        energy_tol_decay: float = 0.7,
        min_energy_tol: float = 5e-5,
        tol_scale: float = 0.02,
        adaptive_iterations: bool = True,
        use_gradient_optimizer: bool = False,
        enable_hamiltonian_batching: bool = False,
    ):
        self.hamiltonian = hamiltonian
        self.optimization_level = optimization_level
        self.max_iterations = int(max_iterations)
        self.seed = seed if seed is not None else 42
        self.random_state = np.random.default_rng(self.seed)

        self.patience = int(max(patience, 1))
        self.window = int(max(window, 1))

        self.ansatz_variant = ansatz_variant
        self.su2_gates = tuple(su2_gates)

        # Hamiltonian batching: distribute Hamiltonian terms across multiple GPUs
        self.enable_hamiltonian_batching = bool(enable_hamiltonian_batching)
        self.num_gpus_detected = self._get_gpu_count()

        self.scale_schedule = list(phi_schedule) if phi_schedule is not None else None
        if scale_config is None:
            self.scale_config = CompoundScaleConfig()
        elif is_dataclass(scale_config):
            self.scale_config = scale_config
        else:
            raise TypeError("scale_config must be a CompoundScaleConfig dataclass or None.")

        self.energy_tol_decay = float(energy_tol_decay)
        self.min_energy_tol = float(min_energy_tol)
        self.tol_scale = float(tol_scale)
        self.base_energy_tol = None if energy_tol is None else float(energy_tol)

        self.default_shots = shots
        self.use_adaptive_iterations = bool(adaptive_iterations)
        self.use_gradient_optimizer = bool(use_gradient_optimizer)

        # Infer number of qubits from the Hamiltonian
        self.num_qubits: int | None = None
        if hasattr(hamiltonian, "num_qubits"):
            self.num_qubits = int(hamiltonian.num_qubits)
        elif hasattr(hamiltonian, "qubit_count"):
            qc_attr = getattr(hamiltonian, "qubit_count")
            self.num_qubits = int(qc_attr() if callable(qc_attr) else qc_attr)
        if self.num_qubits is None:
            raise ValueError("Unable to infer num_qubits from Hamiltonian; please provide one with .num_qubits.")

        # Stage plan: either explicit compound scaling schedule or single-stage config
        self.stage_plan = self._build_stage_plan(
            phi_schedule=self.scale_schedule,
            manual_reps=reps,
            manual_entanglement=entanglement,
            manual_skip=skip_unentangled_qubits,
            manual_shots=shots,
        )

        # Runtime state (populated during execution)
        self.energy_list: List[float] = []
        self.iteration_results: List[Tuple[float, np.ndarray]] = []
        self.stage_offsets: List[int] = []
        self.stage_energy_traces: List[List[float]] = []
        self.stage_summaries: List[dict] = []
        self.cost_history_dict: dict = {}

        # Attributes mutated within stages
        self.kernel = None
        self.param_bounds: List[Tuple[float, float]] | None = None
        self.kernel_param_count: int = 0
        self.shots: int = shots
        self.energy_tol = self.base_energy_tol if self.base_energy_tol is not None else 1e-3

        self._stage_energy: List[float] = []
        self._stage_iteration_results: List[Tuple[float, np.ndarray]] = []
        self._no_improve_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_gpu_count(self) -> int:
        """Get actual GPU count for Hamiltonian batching."""
        try:
            return cudaq.num_available_gpus()
        except AttributeError:
            return 1

    def _build_stage_plan(
        self,
        *,
        phi_schedule: Sequence[int] | None,
        manual_reps: int | None,
        manual_entanglement: EntanglementPattern | None,
        manual_skip: bool | None,
        manual_shots: int,
    ) -> List[dict]:
        if phi_schedule:
            cfg_dict = asdict(self.scale_config)
            plan: List[dict] = []
            for phi in phi_schedule:
                reps, ent, skip, shots = compound_scale(phi, **cfg_dict)
                plan.append(
                    {
                        "phi": phi,
                        "reps": reps,
                        "entanglement": ent,
                        "skip_unentangled_qubits": skip if manual_skip is None else manual_skip,
                        "shots": shots,
                    }
                )
            return plan

        # Single-stage fallback (baseline behaviour)
        plan_reps = manual_reps if manual_reps is not None else 3
        plan_ent = manual_entanglement if manual_entanglement is not None else "reverse_linear"
        plan_skip = manual_skip if manual_skip is not None else False
        return [
            {
                "phi": None,
                "reps": plan_reps,
                "entanglement": plan_ent,
                "skip_unentangled_qubits": plan_skip,
                "shots": manual_shots,
            }
        ]

    def _default_tol_from_shots(self, shots: int) -> float:
        if shots == -1:
            return 1e-4
        if shots <= 0:
            return 1e-4
        return max(1e-4, self.tol_scale / math.sqrt(shots))

    def _select_ansatz_builder(self):
        variant = self.ansatz_variant.lower()
        if variant == "improved":
            return improved_efficient_su2_cudaq
        return efficient_su2_cudaq

    def _reset_logs(self):
        self.energy_list = []
        self.stage_offsets = []
        self.stage_energy_traces = []
        self.stage_summaries = []
        self.iteration_results = []

    def _build_initial_guess(self, prev_params: np.ndarray | None) -> np.ndarray:
        """Fallback positional copy (used when no param_order metadata is available)."""
        if prev_params is None or prev_params.size == 0:
            return self.random_state.random(self.kernel_param_count)

        new_params = np.zeros(self.kernel_param_count, dtype=float)
        copy_len = min(len(prev_params), self.kernel_param_count)
        new_params[:copy_len] = prev_params[:copy_len]
        return new_params

    def _build_semantic_initial_guess(
        self,
        prev_params: np.ndarray | None,
        prev_param_order: List[Tuple] | None,
        new_param_order: List[Tuple],
        prev_reps: int | None,
        new_reps: int,
    ) -> np.ndarray:
        """
        Semantic gate-aware parameter transfer between stages.

        Instead of naive positional copy, matches parameters by their physical
        meaning: (qubit, gate_idx, is_final_layer). This correctly handles:
        - Depth changes (reps increase): final-layer params map to new final layer
        - skip_unentangled changes: params matched by qubit identity, not position
        - New parameters: initialized with small random values (not zero)

        Matching priority:
        1. Exact match: same (qubit, layer, gate_idx) -- direct copy
        2. Final-to-final: old final layer -> new final layer, same (qubit, gate_idx)
        3. No match: small random perturbation U(-0.1, 0.1)

        Args:
            prev_params: Parameter vector from the previous stage (None for first stage)
            prev_param_order: param_order metadata from previous stage
            new_param_order: param_order metadata for the new stage
            prev_reps: reps value from the previous stage
            new_reps: reps value for the new stage

        Returns:
            Initial parameter vector for the new stage
        """
        num_new_params = len(new_param_order)

        # First stage or no previous data: random initialization
        if prev_params is None or prev_params.size == 0 or prev_param_order is None:
            return self.random_state.random(num_new_params)

        # Build lookup from old param_order: (qubit, layer, gate_idx) -> (index, value)
        # Also build final-layer lookup: (qubit, gate_idx) -> (index, value)
        old_exact_lookup: dict[Tuple[int, int, int], float] = {}
        old_final_lookup: dict[Tuple[int, int], float] = {}
        old_final_layer = prev_reps if prev_reps is not None else 0

        for idx, (param_name, qubit, layer, gate_idx) in enumerate(prev_param_order):
            if idx < len(prev_params):
                val = float(prev_params[idx])
                old_exact_lookup[(qubit, layer, gate_idx)] = val
                if layer == old_final_layer:
                    old_final_lookup[(qubit, gate_idx)] = val

        # Map each new parameter
        new_params = np.zeros(num_new_params, dtype=float)
        new_final_layer = new_reps
        matched_exact = 0
        matched_final = 0
        unmatched = 0

        for new_idx, (param_name, qubit, layer, gate_idx) in enumerate(new_param_order):
            key_exact = (qubit, layer, gate_idx)
            key_final = (qubit, gate_idx)

            if key_exact in old_exact_lookup:
                # Priority 1: exact match on (qubit, layer, gate_idx)
                new_params[new_idx] = old_exact_lookup[key_exact]
                matched_exact += 1
            elif layer == new_final_layer and key_final in old_final_lookup:
                # Priority 2: final-to-final mapping (same qubit and gate, both final layers)
                new_params[new_idx] = old_final_lookup[key_final]
                matched_final += 1
            else:
                # Priority 3: no match -- small random perturbation (not zero)
                new_params[new_idx] = self.random_state.uniform(-0.1, 0.1)
                unmatched += 1

        print(f"  Parameter transfer: {matched_exact} exact, {matched_final} final-to-final, {unmatched} new random")
        return new_params

    def _prebuild_all_kernels(self):
        """
        Pre-build all stage kernels and force eager JIT compilation.

        CUDA-Q lazily JIT-compiles kernels at the first cudaq.observe() call.
        If a new kernel is JIT-compiled while another kernel's runtime state is
        active on the mqpu backend, it causes a segfault in the MLIR layer.

        By running a warmup cudaq.observe() on EVERY kernel before the real
        optimization starts, we force all JIT compilation to happen while the
        runtime is clean. Subsequent observe() calls during optimization use
        the cached compiled code without re-triggering JIT.

        Populates self._prebuilt_kernels: list of (kernel, param_order, param_bounds)
        """
        builder = self._select_ansatz_builder()
        self._prebuilt_kernels = []
        for stage_idx, stage_cfg in enumerate(self.stage_plan):
            kernel, param_order, param_bounds = builder(
                num_qubits=self.num_qubits,
                reps=stage_cfg["reps"],
                su2_gates=self.su2_gates,
                entanglement=stage_cfg["entanglement"],
                skip_unentangled_qubits=stage_cfg.get("skip_unentangled_qubits", False),
                skip_final_rotation_layer=False,
            )
            self._prebuilt_kernels.append((kernel, param_order, param_bounds))
            print(f"  Pre-built kernel for stage {stage_idx}: "
                  f"reps={stage_cfg['reps']}, params={len(param_order)}, "
                  f"entanglement={stage_cfg['entanglement']}")

        # Force eager JIT compilation of ALL kernels by running warmup observe
        # calls. This triggers MLIR compilation and GPU memory setup for each
        # kernel while no other kernel's runtime state is active.
        print("  Warming up kernels (forcing JIT compilation)...")
        for stage_idx, (kernel, param_order, _) in enumerate(self._prebuilt_kernels):
            dummy_params = [0.0] * len(param_order)
            try:
                cudaq.observe(kernel, self.hamiltonian, *dummy_params)
                print(f"    Stage {stage_idx} kernel JIT compiled OK ({len(param_order)} params)")
            except Exception as e:
                print(f"    Stage {stage_idx} warmup failed: {e} (will retry at runtime)")
        print("  All kernels ready.")

    def _prepare_stage(self, stage_idx: int, stage_cfg: dict, prev_params: np.ndarray | None) -> np.ndarray:
        # Track stage offset for later analysis
        self.stage_offsets.append(len(self.energy_list))

        # Save previous stage metadata for semantic parameter mapping
        prev_param_order = getattr(self, '_current_param_order', None)
        prev_reps = getattr(self, '_current_reps', None)

        # Use pre-built kernel (avoids CUDA-Q MLIR segfault from JIT-compiling
        # a new kernel after another has already been executed)
        self.kernel, param_order, self.param_bounds = self._prebuilt_kernels[stage_idx]
        self.kernel_param_count = len(param_order)

        # Store current stage metadata for next stage's semantic mapping
        self._current_param_order = param_order
        self._current_reps = stage_cfg["reps"]

        # Update shots and tolerated improvement threshold
        stage_shots = stage_cfg.get("shots", self.default_shots)
        if stage_shots is None:
            stage_shots = self.default_shots
        self.shots = int(stage_shots)

        stage_default_tol = self._default_tol_from_shots(self.shots)
        if stage_idx == 0:
            if self.base_energy_tol is None:
                self.energy_tol = stage_default_tol
            else:
                self.energy_tol = max(self.base_energy_tol, stage_default_tol)
        else:
            decayed = self.energy_tol * self.energy_tol_decay
            self.energy_tol = max(decayed, stage_default_tol, self.min_energy_tol)

        # Reset per-stage buffers
        self.cost_history_dict = {"prev_vector": None, "iters": 0, "cost_history": []}
        self._stage_energy = []
        self._stage_iteration_results = []
        self._no_improve_count = 0

        # Build initial guess using semantic gate-aware mapping
        return self._build_semantic_initial_guess(
            prev_params=prev_params,
            prev_param_order=prev_param_order,
            new_param_order=param_order,
            prev_reps=prev_reps,
            new_reps=stage_cfg["reps"],
        )

    def _stage_callback(self, _xk: np.ndarray):
        n = len(self._stage_energy)
        if n < 2:
            return
        window = min(self.window, n - 1)
        recent = self._stage_energy[-(window + 1) :]
        diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        best_recent_improve = max(diffs) if diffs else float("inf")

        if best_recent_improve < self.energy_tol:
            self._no_improve_count += 1
        else:
            self._no_improve_count = 0

        if not self.use_adaptive_iterations:
            return

        if self._no_improve_count >= self.patience:
            raise CUDAQVQE.EarlyStop()

    def _finalize_stage(self, stage_idx: int, stage_cfg: dict):
        summary = {
            "stage_index": stage_idx,
            "phi": stage_cfg.get("phi"),
            "shots": self.shots,
            "reps": stage_cfg["reps"],
            "entanglement": stage_cfg["entanglement"],
            "skip_unentangled_qubits": stage_cfg.get("skip_unentangled_qubits", False),
            "iterations": len(self._stage_energy),
            "energy_min": min(self._stage_energy) if self._stage_energy else None,
            "energy_final": self._stage_energy[-1] if self._stage_energy else None,
            "adaptive_iterations": self.use_adaptive_iterations,
        }
        self.stage_summaries.append(summary)
        self.stage_energy_traces.append(list(self._stage_energy))

    # ------------------------------------------------------------------
    # Core numeric routines
    # ------------------------------------------------------------------
    def observe_energy(self, params: Iterable[float]) -> float:
        """
        Evaluate expectation value <H> for given parameter vector.
        Handles CUDA-Q observe result across versions.

        If enable_hamiltonian_batching=True and multiple GPUs are available,
        distributes Hamiltonian terms across GPUs for parallel evaluation.
        """
        if isinstance(params, np.ndarray):
            params_list = params.flatten().tolist()
        else:
            params_list = list(params)

        # Determine if we should use Hamiltonian batching
        use_batching = (
            self.enable_hamiltonian_batching
            and self.num_gpus_detected > 1
        )

        try:
            if use_batching:
                # Hamiltonian batching: distribute terms across GPUs
                # Uses execution=cudaq.parallel.thread to enable multi-GPU distribution
                try:
                    if self.shots == -1:
                        obs_result = cudaq.observe(
                            self.kernel, self.hamiltonian, *params_list,
                            execution=cudaq.parallel.thread
                        )
                    else:
                        obs_result = cudaq.observe(
                            self.kernel, self.hamiltonian, *params_list,
                            shots_count=self.shots,
                            execution=cudaq.parallel.thread
                        )
                except (TypeError, AttributeError):
                    # Fallback if execution parameter not supported
                    if self.shots == -1:
                        obs_result = cudaq.observe(self.kernel, self.hamiltonian, *params_list)
                    else:
                        obs_result = cudaq.observe(
                            self.kernel, self.hamiltonian, *params_list,
                            shots_count=self.shots
                        )
            else:
                # Standard sequential execution
                if self.shots == -1:
                    obs_result = cudaq.observe(self.kernel, self.hamiltonian, *params_list)
                else:
                    obs_result = cudaq.observe(self.kernel, self.hamiltonian, *params_list, shots_count=self.shots)
        except TypeError:
            if self.shots == -1:
                obs_result = cudaq.observe(self.kernel, self.hamiltonian, params_list)
            else:
                obs_result = cudaq.observe(self.kernel, self.hamiltonian, params_list, shots_count=self.shots)

        if hasattr(obs_result, "expectation_value"):
            exp_attr = getattr(obs_result, "expectation_value")
            return float(exp_attr() if callable(exp_attr) else exp_attr)
        if hasattr(obs_result, "expectation"):
            exp_attr = getattr(obs_result, "expectation")
            return float(exp_attr() if callable(exp_attr) else exp_attr)
        if hasattr(obs_result, "energy"):
            energy_attr = getattr(obs_result, "energy")
            return float(energy_attr() if callable(energy_attr) else energy_attr)
        return float(obs_result)

    def cost_func(self, params: np.ndarray) -> float:
        t0 = time.perf_counter()
        energy = self.observe_energy(params)
        dt = time.perf_counter() - t0
        self.energy_list.append(energy)
        self._stage_energy.append(energy)
        params_copy = np.array(params, dtype=float)
        self._stage_iteration_results.append((energy, params_copy))

        self.cost_history_dict["iters"] += 1
        self.cost_history_dict["prev_vector"] = params_copy
        self.cost_history_dict["cost_history"].append(energy)
        print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {energy}]  ({dt:.2f}s)")
        return energy

    def run_vqe(self, init_params: Iterable[float] | None = None, phi_schedule: Sequence[int] | None = None):
        """
        Executes VQE with optional staged scaling.

        Returns the global best result across ALL stages (not just the final one),
        ensuring that good solutions found in earlier stages are never lost.

        Returns
        -------
        energy_list : List[float]
            Energies collected across all stages (in chronological order).
        best_params : np.ndarray
            Best parameters found across all stages (paired with best_kernel).
        kernel : cudaq.Kernel
            The CUDA-Q kernel that produced the best energy (from any stage).
        top_results : List[Tuple[float, np.ndarray]]
            Top-6 (energy, params) tuples from the stage that produced the best energy.
        """
        self._reset_logs()

        if phi_schedule is not None:
            # Override plan with user-supplied schedule
            self.stage_plan = self._build_stage_plan(
                phi_schedule=phi_schedule,
                manual_reps=None,
                manual_entanglement=None,
                manual_skip=None,
                manual_shots=self.default_shots,
            )

        prev_params = None if init_params is None else np.array(init_params, dtype=float)
        best_params = prev_params

        # Pre-build ALL stage kernels before any execution to avoid CUDA-Q
        # MLIR segfaults from JIT-compiling after execution has started.
        print("[Pre-building kernels for all stages]")
        self._prebuild_all_kernels()

        # Global best tracking across ALL stages
        global_best_energy = float("inf")
        global_best_params = None
        global_best_stage_idx = -1
        # Per-stage top results: stage_idx -> sorted list of (energy, params)
        stage_top_results: dict[int, List[Tuple[float, np.ndarray]]] = {}

        for stage_idx, stage_cfg in enumerate(self.stage_plan):
            stage_start = time.perf_counter()
            x0 = self._prepare_stage(stage_idx, stage_cfg, prev_params)

            try:
                if self.use_gradient_optimizer:
                    # Use gradient-based optimizer (L-BFGS-B with parameter-shift gradients)
                    print(f"[Stage {stage_idx}] Using gradient-based optimizer (L-BFGS-B + parameter-shift)")
                    grad_optimizer = GradientBasedVQEOptimizer(
                        kernel=self.kernel,
                        hamiltonian=self.hamiltonian,
                        shots=self.shots,
                        max_iterations=self.max_iterations,
                        gtol=self.energy_tol * 0.1,  # Gradient tolerance
                        ftol=self.energy_tol,  # Function tolerance
                        energy_tol=self.energy_tol,  # Adaptive stopping: same as COBYLA
                        patience=self.patience,  # Adaptive stopping: same as COBYLA
                        window=self.window,  # Adaptive stopping: same as COBYLA
                        use_adaptive_iterations=self.use_adaptive_iterations,  # Adaptive stopping flag
                        enable_hamiltonian_batching=self.enable_hamiltonian_batching,  # Multi-GPU Hamiltonian distribution
                    )
                    stage_best_params, final_energy = grad_optimizer.optimize(
                        initial_params=x0,
                        bounds=self.param_bounds,
                    )
                    # Update tracking from gradient optimizer
                    self.energy_list.extend(grad_optimizer.energy_history)
                    self._stage_energy.extend(grad_optimizer.energy_history)
                    # Use actual parameter history instead of final params as proxy
                    for energy, params in zip(grad_optimizer.energy_history, grad_optimizer.params_history):
                        self._stage_iteration_results.append((energy, params))
                    stage_best_params = np.array(stage_best_params, dtype=float)
                else:
                    # Use derivative-free optimizer (COBYLA)
                    print(f"[Stage {stage_idx}] Using derivative-free optimizer (COBYLA)")
                    callback = self._stage_callback if self.use_adaptive_iterations else None
                    res = minimize(
                        self.cost_func,
                        x0,
                        method="cobyla",
                        options={"maxiter": self.max_iterations},
                        callback=callback,
                    )
                    stage_best_params = np.array(res.x, dtype=float)
            except CUDAQVQE.EarlyStop:
                if not self._stage_iteration_results:
                    raise RuntimeError("Early stop triggered without any stage iterations recorded.")
                stage_best_energy, stage_best_params = min(self._stage_iteration_results, key=lambda t: t[0])
                stage_best_params = np.array(stage_best_params, dtype=float)
            except Exception:
                # Re-raise but keep stage buffers for debugging
                raise

            # Extract the BEST params from this stage (lowest energy), not just
            # the final iteration params which may be worse due to optimizer overshoot.
            if self._stage_iteration_results:
                sorted_stage = sorted(self._stage_iteration_results, key=lambda t: t[0])
                stage_best_energy_val, stage_true_best_params = sorted_stage[0]
                stage_true_best_params = np.array(stage_true_best_params, dtype=float)
                print(f"  Stage {stage_idx} best energy from history: {stage_best_energy_val:.6f}")

                # Use the TRUE best params to seed the next stage
                prev_params = stage_true_best_params
                best_params = stage_true_best_params

                stage_top_results[stage_idx] = sorted_stage[:6]

                # Update global best if this stage found a better energy
                if stage_best_energy_val < global_best_energy:
                    global_best_energy = stage_best_energy_val
                    global_best_params = stage_true_best_params.copy()
                    global_best_stage_idx = stage_idx
                    print(f"  [Global best] Stage {stage_idx} achieved new global best: {global_best_energy:.6f}")
            else:
                # No iteration results recorded (shouldn't happen, but be safe)
                prev_params = stage_best_params
                best_params = stage_best_params

            # Accumulate iteration results across all stages
            self.iteration_results.extend(list(self._stage_iteration_results))

            stage_elapsed = time.perf_counter() - stage_start
            n_iters = len(self._stage_iteration_results)
            avg_iter = stage_elapsed / n_iters if n_iters > 0 else 0.0
            print(f"  Stage {stage_idx} wall time: {stage_elapsed:.1f}s "
                  f"({n_iters} iters, avg {avg_iter:.2f}s/iter)")

            self._finalize_stage(stage_idx, stage_cfg)

        # Return the global best result from whichever stage found it
        if global_best_params is not None:
            best_params = global_best_params
            top_results = stage_top_results.get(global_best_stage_idx, [])
            # Use the pre-built kernel for the best stage
            best_kernel = self._prebuilt_kernels[global_best_stage_idx][0]
            print(f"\n  Global best energy {global_best_energy:.6f} from stage {global_best_stage_idx}")
        else:
            # Fallback: use final stage results
            best_kernel = self.kernel
            top_results = stage_top_results.get(len(self.stage_plan) - 1, [])

        return self.energy_list, best_params, best_kernel, top_results


__all__ = ["CUDAQVQE", "qiskit_sparsepauliop_to_cudaq_spin"]
