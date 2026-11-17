"""
ADAPT-VQE: Adaptive Derivative-Assembled Pseudo-Trotter ansatz VQE.

Grows the ansatz circuit one operator at a time by selecting at each step
the operator from a fixed pool that has the largest energy gradient.
Converges when the maximum gradient across all pool operators falls below
gradient_threshold (original stopping criterion from Grimsley et al. 2019).

Reference: Grimsley et al. (2019) "An adaptive variational algorithm for
exact molecular simulations on a quantum computer" — Nature Commun. 10, 3007.

Operator pool (hardware-efficient, entanglement-agnostic):
  - Ry(θ) on each qubit            (single-qubit rotations)
  - Rz(θ) on each qubit
  - RZZ(θ) on each entanglement pair  (exp(-iθ ZZ) via CX–Rz–CX)

Initialization: Hadamard on all qubits → |+⟩^n reference state.
Inner optimizer: COBYLA (derivative-free).
Outer stopping: gradient threshold on pool gradients.
"""

from __future__ import annotations

import time
from math import pi
from typing import List, Optional, Tuple

import cudaq
import numpy as np
from scipy.optimize import minimize

from .efficientsu2 import pairs_for_pattern


# Type alias: (gate_name, qubit_a, qubit_b_or_None)
Op = Tuple[str, int, Optional[int]]


# ------------------------------------------------------------------
# Kernel factory
# ------------------------------------------------------------------

def _make_adapt_kernel(n_qubits: int, ops: List[Op]):
    """
    Build a CUDA-Q kernel for the current ADAPT ansatz.

    Prepends Hadamard on all qubits (|+>^n reference state), then
    applies the selected ops in order.

    ops: ordered list of (gate, q0, q1_or_None) already selected.
    Returns (kernel, n_params).

    Gates supported:
      'ry'  – Ry(theta) on q0
      'rz'  – Rz(theta) on q0
      'rzz' – exp(-i*theta*ZZ) on (q0,q1): CX, Rz(2*theta), CX
    """
    n = len(ops)
    if n == 0:
        mk = cudaq.make_kernel()
        if isinstance(mk, tuple):
            kernel = mk[0]
        else:
            kernel = mk
        qvec = kernel.qalloc(n_qubits)
        for i in range(n_qubits):
            kernel.h(qvec[i])
        return kernel, 0

    mk = cudaq.make_kernel(*([float] * n))
    if isinstance(mk, tuple):
        kernel = mk[0]
        thetas = list(mk[1:])
    else:
        kernel = mk
        thetas = [kernel.argument(i) for i in range(n)]

    qvec = kernel.qalloc(n_qubits)

    # Hadamard initialization: |+>^n
    for i in range(n_qubits):
        kernel.h(qvec[i])

    for i, (gate, q0, q1) in enumerate(ops):
        t = thetas[i]
        if gate == "ry":
            kernel.ry(t, qvec[q0])
        elif gate == "rz":
            kernel.rz(t, qvec[q0])
        elif gate == "rzz":
            # exp(-i*t*Z⊗Z) = CX · Rz(2t) · CX
            kernel.cx(qvec[q0], qvec[q1])
            kernel.rz(t, qvec[q1])
            kernel.cx(qvec[q0], qvec[q1])

    return kernel, n


# ------------------------------------------------------------------
# ADAPT-VQE class
# ------------------------------------------------------------------

class AdaptVQE:
    """
    ADAPT-VQE optimizer for CUDA-Q + tensornet-mps-mgpu backend.

    Original algorithm (Grimsley et al. 2019):
    - Start from |+>^n (Hadamard on all qubits)
    - Each step: select operator with max gradient from pool (reuse allowed)
    - Inner optimization: COBYLA (derivative-free)
    - Outer stopping: gradient threshold OR eigenvalue change threshold
    """

    def __init__(
        self,
        hamiltonian,
        num_qubits: int,
        entanglement: str = "circular",
        reps_for_pool: int = 4,
        gradient_threshold: float = 1e-5,
        eigenvalue_threshold: float = 1e-5,
        max_iterations: Optional[int] = None,
        max_iter: int = 1000000,
        energy_tol: float = 1e-4,
        patience: int = 12,
        window: int = 40,
        shots: int = -1,
        seed: int = 0,
    ):
        """
        Args:
            hamiltonian: CUDA-Q SpinOperator.
            num_qubits: Circuit width.
            entanglement: Entanglement pattern for the operator pool.
            reps_for_pool: Number of EfficientSU2 layers for RZZ pool pairs.
            gradient_threshold: Outer stopping: stop when max pool gradient < this.
            eigenvalue_threshold: Outer stopping: stop when energy change between
                consecutive ADAPT steps < this (matches Qiskit AdaptVQE default 1e-5).
            max_iterations: Hard cap on ADAPT steps (None = unlimited).
            max_iter: Max COBYLA iterations per ADAPT step.
            energy_tol: Early-stop tolerance for inner COBYLA.
            patience: Early-stop patience for inner COBYLA.
            window: Early-stop window for inner COBYLA.
            shots: -1 for analytic (statevector) observe, >0 for shot-based.
            seed: RNG seed for initial parameters.
        """
        self.hamiltonian = hamiltonian
        self.num_qubits = int(num_qubits)
        self.entanglement = entanglement
        self.gradient_threshold = float(gradient_threshold)
        self.eigenvalue_threshold = float(eigenvalue_threshold)
        self.max_iterations = max_iterations
        self.max_iter = int(max_iter)
        self.energy_tol = float(energy_tol)
        self.patience = int(patience)
        self.window = int(window)
        self.shots = int(shots)
        self.seed = seed

        # Build operator pool: Ry/Rz per qubit + RZZ per entanglement pair
        self._pool: List[Op] = self._build_pool(reps_for_pool)

        # State
        self.selected_ops: List[Op] = []
        self.params: np.ndarray = np.array([], dtype=float)
        self.energy_history: List[float] = []
        self.nfev: int = 0
        self.adapt_steps: int = 0

        # Metadata (mirroring CUDAQVQE fields used by Main.py)
        self.shots_final = shots
        self.stage_plan = [{"phi": 0}]
        self.stage_offsets = [0]
        self.stage_summaries: List[dict] = []
        self.use_adaptive_iterations = True
        self.max_epochs = max_iterations or 1000000

    # ------------------------------------------------------------------

    def _build_pool(self, reps_for_pool: int) -> List[Op]:
        """Ry/Rz on each qubit + RZZ on entanglement pairs across reps_for_pool layers."""
        pool: List[Op] = []
        seen_pairs = set()
        for q in range(self.num_qubits):
            pool.append(("ry", q, None))
            pool.append(("rz", q, None))
        for ell in range(reps_for_pool):
            for c, t in pairs_for_pattern(self.num_qubits, ell, self.entanglement):
                key = (min(c, t), max(c, t))
                if key not in seen_pairs:
                    pool.append(("rzz", c, t))
                    seen_pairs.add(key)
        return pool

    # ------------------------------------------------------------------

    def _observe(self, kernel, params: List[float]) -> float:
        """Evaluate <H> for given kernel and params."""
        self.nfev += 1
        if self.shots == -1:
            result = cudaq.observe(kernel, self.hamiltonian, *params)
        else:
            result = cudaq.observe(kernel, self.hamiltonian, *params,
                                   shots_count=self.shots)
        if hasattr(result, "expectation_value"):
            return float(result.expectation_value)
        if hasattr(result, "expectation"):
            exp = result.expectation
            return float(exp() if callable(exp) else exp)
        return float(result)

    def _candidate_gradient(self, op: Op) -> float:
        """
        Gradient of appending `op` to the current ansatz at theta=0,
        estimated via the parameter-shift rule on a trial circuit.
        """
        trial_ops = self.selected_ops + [op]
        kernel, n = _make_adapt_kernel(self.num_qubits, trial_ops)
        if n == 0:
            return 0.0
        shift = pi / 2
        params_plus = list(self.params) + [shift]
        params_minus = list(self.params) + [-shift]
        e_plus = self._observe(kernel, params_plus)
        e_minus = self._observe(kernel, params_minus)
        return abs(e_plus - e_minus) / 2.0

    # ------------------------------------------------------------------

    def _cobyla_optimize(self, kernel, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run COBYLA on the current ansatz with early stopping (mirrors CUDAQVQE)."""
        class EarlyStop(Exception):
            pass

        energy_log: List[float] = []
        history: List[Tuple[float, np.ndarray]] = []
        no_improve_count = [0]

        def cost(params):
            e = self._observe(kernel, params.tolist())
            energy_log.append(e)
            history.append((e, params.copy()))
            return e

        def callback(_xk):
            n = len(energy_log)
            if n < 2:
                return
            window = min(self.window, n - 1)
            recent = energy_log[-(window + 1):]
            diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
            best_improve = max(diffs) if diffs else float("inf")
            if best_improve < self.energy_tol:
                no_improve_count[0] += 1
            else:
                no_improve_count[0] = 0
            if no_improve_count[0] >= self.patience:
                raise EarlyStop()

        try:
            minimize(cost, x0, method="cobyla",
                     options={"maxiter": self.max_iter},
                     callback=callback)
        except EarlyStop:
            pass

        if not history:
            return x0.copy(), float("inf")

        self.energy_history.extend(energy_log)
        best_e, best_p = min(history, key=lambda t: t[0])
        return np.array(best_p, dtype=float), float(best_e)

    # ------------------------------------------------------------------

    def run_vqe(self) -> Tuple[List[float], np.ndarray, object, List]:
        """
        Run ADAPT-VQE.

        Returns:
            energy_list: All energies observed during optimization.
            best_params: Best parameter vector found.
            kernel: Final CUDA-Q ansatz kernel.
            top_results: List of (energy, params) top-6 pairs.
        """
        np.random.seed(self.seed)
        self.selected_ops = []
        self.params = np.array([], dtype=float)
        self.energy_history = []
        self.nfev = 0
        self.adapt_steps = 0

        best_energy = float("inf")
        best_params = np.array([], dtype=float)
        best_ops: List[Op] = []
        top_results: List[Tuple[float, np.ndarray]] = []

        print(f"\n{'='*60}")
        print(f"ADAPT-VQE (COBYLA inner, gradient-threshold + eigenvalue-threshold outer)")
        print(f"{'='*60}")
        print(f"Qubits: {self.num_qubits}  |  Pool size: {len(self._pool)} (reuse allowed)")
        print(f"gradient_threshold: {self.gradient_threshold:.1e}  |  "
              f"eigenvalue_threshold: {self.eigenvalue_threshold:.1e}  |  "
              f"max_iterations: {self.max_iterations}")
        print(f"Initialization: |+>^n (Hadamard on all qubits)")

        prev_energy = float("inf")
        step = 0
        no_improve = 0  # patience counter across ADAPT steps
        while True:
            if self.max_iterations is not None and step >= self.max_iterations:
                print(f"\nADAPT stopped: max_iterations={self.max_iterations} reached")
                break

            # Compute gradient for every pool operator (reuse allowed — full pool each step)
            print(f"\n--- ADAPT step {step + 1}: evaluating {len(self._pool)} candidates ---")
            gradients = [self._candidate_gradient(op) for op in self._pool]

            max_grad = max(gradients)
            best_op_idx = int(np.argmax(gradients))
            best_op = self._pool[best_op_idx]

            print(f"  Max gradient: {max_grad:.6e}  (op: {best_op})")

            # Outer stopping 1: gradient threshold (Grimsley et al.)
            if max_grad < self.gradient_threshold:
                print(f"\nADAPT converged: max gradient {max_grad:.2e} < "
                      f"gradient_threshold {self.gradient_threshold:.2e}")
                break

            # Append best operator (same op can appear multiple times)
            self.selected_ops.append(best_op)

            # Extend params: new op starts at 0
            self.params = np.append(self.params, 0.0)
            kernel, n_params = _make_adapt_kernel(self.num_qubits, self.selected_ops)

            print(f"  Appended op {best_op}. Ansatz size: {len(self.selected_ops)} ops")
            print(f"  Optimising {n_params} parameters with COBYLA ...")

            self.params, energy = self._cobyla_optimize(kernel, self.params)

            print(f"  Energy after step {step + 1}: {energy:.6f}")

            # Outer stopping 2: eigenvalue threshold (Qiskit AdaptVQE criterion)
            energy_change = abs(prev_energy - energy)
            if step > 0 and energy_change < self.eigenvalue_threshold:
                print(f"\nADAPT converged: energy change {energy_change:.2e} < "
                      f"eigenvalue_threshold {self.eigenvalue_threshold:.2e}")
                if energy < best_energy:
                    best_energy = energy
                    best_params = self.params.copy()
                    best_ops = list(self.selected_ops)
                top_results.append((energy, self.params.copy()))
                self.adapt_steps += 1
                break
            prev_energy = energy

            if energy < best_energy:
                best_energy = energy
                best_params = self.params.copy()
                best_ops = list(self.selected_ops)
                no_improve = 0
                print(f"  New best: {best_energy:.6f}")
            else:
                no_improve += 1
                print(f"  No improvement ({no_improve}/{self.patience})")
                if no_improve >= self.patience:
                    print(f"\nADAPT converged: no energy improvement > {self.energy_tol:.1e} "
                          f"for {self.patience} consecutive steps")
                    break

            top_results.append((energy, self.params.copy()))
            self.adapt_steps += 1
            step += 1

        print(f"\n{'='*60}")
        print(f"ADAPT-VQE Complete")
        print(f"{'='*60}")
        print(f"ADAPT steps: {self.adapt_steps}")
        print(f"Total function evaluations: {self.nfev}")
        print(f"Final ansatz size: {len(self.selected_ops)} operators")
        print(f"Best energy: {best_energy:.6f}")
        print(f"{'='*60}\n")

        if best_ops:
            best_kernel, _ = _make_adapt_kernel(self.num_qubits, best_ops)
        else:
            best_ops = list(self.selected_ops)
            best_kernel, _ = _make_adapt_kernel(self.num_qubits, best_ops)
            if not best_params.size:
                best_params = np.zeros(len(best_ops))

        n_best = len(best_ops)
        top_results_filtered = [(e, p) for e, p in top_results if len(p) == n_best]
        top_results_sorted = sorted(
            top_results_filtered if top_results_filtered else [(best_energy, best_params)],
            key=lambda x: x[0],
        )[:6]

        self.stage_summaries = [{
            "stage_index": 0,
            "phi": 0,
            "shots": self.shots,
            "reps": len(self.selected_ops),
            "entanglement": self.entanglement,
            "skip_unentangled_qubits": False,
            "iterations": self.nfev,
            "energy_min": best_energy,
            "energy_final": best_energy,
            "adaptive_iterations": True,
        }]

        return self.energy_history, best_params, best_kernel, top_results_sorted
