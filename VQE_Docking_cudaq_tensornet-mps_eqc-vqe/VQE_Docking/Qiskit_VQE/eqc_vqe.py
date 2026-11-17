"""
EQC-VQE: Ensembled Quantum Computing VQE.

Core idea from the paper: instead of COBYLA (derivative-free), use
parameter-shift gradient descent (ASGD) where each gradient is computed
for one parameter at a time and workers run asynchronously across QPUs.

In our noiseless tensornet-mps-mgpu simulation:
  - P_Correct = 1.0 for all workers (no device noise to weight)
  - The ASGD update simplifies to standard gradient descent:
      θ_i ← θ_i − α * (∂E/∂θ_i)
  - "Workers" are ThreadPoolExecutor threads computing per-parameter
    gradients in parallel, mirroring the paper's multi-QPU parallelism.

Gradient via parameter-shift rule (Equation from paper's Algorithm 2):
    ∂E/∂θ_i = [E(θ_i + π/2) − E(θ_i − π/2)] / 2

Reference: Stein et al. (2021) "EQC: Ensembled Quantum Computing for
Variational Quantum Algorithms" arXiv:2111.14940
  learning_rate α = 0.1  (Section V-B)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import pi
from typing import List, Optional, Tuple

import cudaq
import numpy as np

from .efficientsu2 import efficient_su2_cudaq


class EQCVQE:
    """
    EQC-VQE optimizer using parameter-shift gradient descent.

    Replaces COBYLA with analytic gradient descent.  Each epoch computes
    the full gradient vector via parameter-shift in parallel (one thread
    per parameter, matching EQC's one-worker-per-parameter parallelism)
    then applies a single gradient-descent step.

    Stopping criterion: same energy-improvement patience rule as
    baseline/CVaR/ADAPT — stops when energy improvement < energy_tol
    for `patience` consecutive epochs.
    """

    def __init__(
        self,
        hamiltonian,
        num_qubits: int,
        entanglement: str = "circular",
        reps: int = 4,
        learning_rate: float = 0.1,
        max_epochs: int = 1000000,
        shots: int = 100000,
        seed: int = 0,
        energy_tol: float = 1e-4,
        patience: int = 12,
        window: int = 40,
        num_workers: Optional[int] = None,
        skip_unentangled_qubits: bool = False,
    ):
        """
        Args:
            hamiltonian: CUDA-Q SpinOperator.
            num_qubits: Circuit width.
            entanglement: EfficientSU2 entanglement pattern.
            reps: EfficientSU2 repetition depth.
            learning_rate: Gradient descent step size α (paper default 0.1).
            max_epochs: Hard cap on epochs (effectively unlimited with early stopping).
            shots: Passed to cudaq.observe; ignored by tensornet-mps-mgpu (analytic).
            seed: RNG seed for initial parameters.
            energy_tol: Energy improvement threshold for early stopping.
            patience: Consecutive epochs without improvement before stopping.
            window: Window size for improvement tracking.
            num_workers: Parallel threads for gradient computation.
                         Defaults to min(n_params, 8).
            skip_unentangled_qubits: Passed to EfficientSU2.
        """
        self.hamiltonian = hamiltonian
        self.num_qubits = int(num_qubits)
        self.entanglement = entanglement
        self.reps = int(reps)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.shots = int(shots)
        self.seed = int(seed)
        self.energy_tol = float(energy_tol)
        self.patience = int(patience)
        self.window = int(window)
        self.skip_unentangled_qubits = skip_unentangled_qubits

        # Build ansatz kernel (same EfficientSU2 as baseline)
        self.kernel, self.param_order, self.bounds = efficient_su2_cudaq(
            num_qubits=self.num_qubits,
            reps=self.reps,
            su2_gates=("ry", "rz"),
            entanglement=self.entanglement,
            skip_unentangled_qubits=self.skip_unentangled_qubits,
        )
        self.n_params = len(self.param_order)
        self.num_workers = min(self.n_params, num_workers or 8)

        # State tracking (compatible with Main.py metadata)
        self.energy_history: List[float] = []
        self.nfev: int = 0

        # CUDAQVQE-compatible metadata fields
        self.stage_plan = [{"phi": 0}]
        self.stage_offsets = [0]
        self.stage_summaries: List[dict] = []
        self.use_adaptive_iterations = True

    # ------------------------------------------------------------------

    def _observe(self, params: np.ndarray) -> float:
        """Evaluate <H> analytically (shots ignored by backend)."""
        self.nfev += 1
        result = cudaq.observe(self.kernel, self.hamiltonian, *params.tolist())
        if hasattr(result, "expectation_value"):
            return float(result.expectation_value)
        exp = result.expectation
        return float(exp() if callable(exp) else exp)

    def _param_shift_gradient(self, params: np.ndarray, i: int) -> float:
        """
        Parameter-shift gradient for parameter i:
            g_i = [E(θ_i + π/2) − E(θ_i − π/2)] / 2
        P_Correct = 1 (noiseless simulation), so no weighting applied.
        """
        p_plus = params.copy()
        p_plus[i] += pi / 2
        p_minus = params.copy()
        p_minus[i] -= pi / 2
        e_plus = self._observe(p_plus)
        e_minus = self._observe(p_minus)
        return (e_plus - e_minus) / 2.0

    def _compute_gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute full gradient vector in parallel across parameters.
        Each thread handles one parameter (mirrors EQC's one-worker-per-param).
        """
        gradient = np.zeros(self.n_params)
        lock = threading.Lock()

        def worker(i):
            g = self._param_shift_gradient(params, i)
            return i, g

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(worker, i): i for i in range(self.n_params)}
            for future in as_completed(futures):
                i, g = future.result()
                with lock:
                    gradient[i] = g

        return gradient

    # ------------------------------------------------------------------

    def run_vqe(self) -> Tuple[List[float], np.ndarray, object, List]:
        """
        Run EQC-VQE: gradient descent with parameter-shift gradients.

        Returns:
            energy_history: Energy after each epoch.
            best_params: Parameters giving lowest energy.
            kernel: EfficientSU2 ansatz kernel.
            top_results: Top-6 (energy, params) pairs sorted by energy.
        """
        np.random.seed(self.seed)
        params = np.random.uniform(-pi, pi, self.n_params)

        self.energy_history = []
        self.nfev = 0
        best_energy = float("inf")
        best_params = params.copy()
        top_results: List[Tuple[float, np.ndarray]] = []

        no_improve = 0

        print(f"\n{'='*60}")
        print(f"EQC-VQE  (α={self.learning_rate}, workers={self.num_workers})")
        print(f"{'='*60}")
        print(f"Qubits: {self.num_qubits}  |  Params: {self.n_params}  |  "
              f"Entanglement: {self.entanglement}  |  Reps: {self.reps}")

        for epoch in range(1, self.max_epochs + 1):
            # Compute full gradient in parallel
            gradient = self._compute_gradient(params)

            # Gradient descent update: θ ← θ − α * g  (P_Correct = 1)
            params = params - self.learning_rate * gradient

            # Evaluate energy at updated params
            energy = self._observe(params)
            self.energy_history.append(energy)
            top_results.append((energy, params.copy()))

            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}: E = {energy:.6f}  |  "
                      f"|grad| = {np.linalg.norm(gradient):.4e}")

            # Early stopping: window-based, identical to baseline/CVaR/ADAPT.
            # Check best improvement over last `window` epochs; if < energy_tol
            # for `patience` consecutive checks, stop.
            n = len(self.energy_history)
            if n >= 2:
                w = min(self.window, n - 1)
                recent = self.energy_history[-(w + 1):]
                best_imp = max(recent[i - 1] - recent[i]
                               for i in range(1, len(recent)))
                if best_imp < self.energy_tol:
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve >= self.patience:
                    print(f"\nEQC-VQE converged at epoch {epoch}: "
                          f"best improvement over last {w} epochs < "
                          f"{self.energy_tol:.1e} for {self.patience} "
                          f"consecutive checks")
                    break

        print(f"\n{'='*60}")
        print(f"EQC-VQE Complete")
        print(f"{'='*60}")
        print(f"Epochs: {len(self.energy_history)}")
        print(f"Total function evaluations: {self.nfev}")
        print(f"Best energy: {best_energy:.6f}")
        print(f"{'='*60}\n")

        top_results_sorted = sorted(top_results, key=lambda x: x[0])[:6]

        self.stage_summaries = [{
            "stage_index": 0,
            "phi": 0,
            "shots": self.shots,
            "reps": self.reps,
            "entanglement": self.entanglement,
            "skip_unentangled_qubits": self.skip_unentangled_qubits,
            "iterations": len(self.energy_history),
            "energy_min": best_energy,
            "energy_final": float(self.energy_history[-1]) if self.energy_history else best_energy,
            "adaptive_iterations": True,
            "learning_rate": self.learning_rate,
        }]

        return self.energy_history, best_params, self.kernel, top_results_sorted
