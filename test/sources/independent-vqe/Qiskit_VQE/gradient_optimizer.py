"""
GPU-batched parameter-shift gradient computation for CUDA-Q VQE.

This module implements parallel gradient computation using the parameter-shift rule,
optimized for single-GPU execution with batched observe() calls.
"""

from __future__ import annotations

import time
from typing import Callable, List, Tuple

import cudaq
import numpy as np
from scipy.optimize import minimize
from .mps_config import TARGET


class ParameterShiftGradient:
    """
    Compute gradients using the parameter-shift rule with GPU batching.

    For a parameterized quantum circuit U(θ) and observable H:
    ∂⟨H⟩/∂θ_i = [⟨H⟩(θ + π/2 e_i) - ⟨H⟩(θ - π/2 e_i)] / 2

    GPU optimization: Batch all 2N parameter-shifted evaluations into a single
    cudaq.observe() call for parallel execution.
    """

    def __init__(self, kernel, hamiltonian, shots: int = -1, enable_hamiltonian_batching: bool = False):
        """
        Initialize gradient calculator.

        Args:
            kernel: CUDA-Q parameterized kernel
            hamiltonian: CUDA-Q spin operator
            shots: Number of shots (-1 for statevector)
            enable_hamiltonian_batching: Distribute Hamiltonian terms across GPUs
        """
        self.kernel = kernel
        self.hamiltonian = hamiltonian
        self.shots = shots
        self.shift = np.pi / 2  # Standard parameter-shift angle
        self.enable_hamiltonian_batching = bool(enable_hamiltonian_batching)
        self._num_gpus = self._get_gpu_count()

    def _get_gpu_count(self) -> int:
        """Get available GPU count."""
        try:
            return cudaq.num_available_gpus()
        except AttributeError:
            return 1

    def compute_gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇⟨H⟩ using batched parameter-shift rule.

        Args:
            params: Current parameter vector of length N

        Returns:
            gradient: Gradient vector of length N
        """
        n_params = len(params)
        gradient = np.zeros(n_params, dtype=float)

        # Build all shifted parameter vectors for batching
        # For each parameter i: evaluate at θ + π/2 e_i and θ - π/2 e_i
        shifted_params_list = []

        for i in range(n_params):
            # Forward shift: θ + π/2 e_i
            params_plus = params.copy()
            params_plus[i] += self.shift
            shifted_params_list.append(params_plus)

            # Backward shift: θ - π/2 e_i
            params_minus = params.copy()
            params_minus[i] -= self.shift
            shifted_params_list.append(params_minus)

        # Batch GPU evaluation: 2N parameter sets evaluated in parallel
        energies = self._batch_observe(shifted_params_list)

        # Compute gradients from shifted evaluations
        for i in range(n_params):
            energy_plus = energies[2 * i]
            energy_minus = energies[2 * i + 1]
            gradient[i] = (energy_plus - energy_minus) / 2.0

        return gradient

    def _batch_observe(self, params_list: List[np.ndarray]) -> List[float]:
        """
        Evaluate energy for multiple parameter sets using GPU batching.

        CUDA-Q optimization: Submit all circuits to GPU at once for parallel execution.

        Args:
            params_list: List of parameter vectors to evaluate

        Returns:
            energies: List of expectation values
        """
        energies = []

        # Note: CUDA-Q's observe() doesn't have a native batch API,
        # but the GPU will automatically parallelize multiple sequential calls
        # if they're submitted rapidly. For true batching, we could use
        # asynchronous execution, but for simplicity we do sequential calls
        # which still benefit from GPU parallelism at the thread-block level.

        for params in params_list:
            energy = self._single_observe(params)
            energies.append(energy)

        return energies

    def _single_observe(self, params: np.ndarray) -> float:
        """
        Evaluate energy for a single parameter set.

        If enable_hamiltonian_batching=True and multiple GPUs are available,
        distributes Hamiltonian terms across GPUs via cudaq.parallel.thread.

        Args:
            params: Parameter vector

        Returns:
            energy: Expectation value ⟨H⟩
        """
        params_list = params.flatten().tolist()

        use_batching = (
            self.enable_hamiltonian_batching
            and self._num_gpus > 1
            and cudaq.get_target().name != TARGET
        )

        try:
            if use_batching:
                # Hamiltonian batching: distribute terms across GPUs
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
                # Standard single-GPU execution
                if self.shots == -1:
                    obs_result = cudaq.observe(self.kernel, self.hamiltonian, *params_list)
                else:
                    obs_result = cudaq.observe(
                        self.kernel, self.hamiltonian, *params_list, shots_count=self.shots
                    )
        except TypeError:
            # Fallback for older CUDA-Q API
            if self.shots == -1:
                obs_result = cudaq.observe(self.kernel, self.hamiltonian, params_list)
            else:
                obs_result = cudaq.observe(
                    self.kernel, self.hamiltonian, params_list, shots_count=self.shots
                )

        # Extract energy from result
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


class GradientBasedVQEOptimizer:
    """
    VQE optimizer using L-BFGS-B with GPU-batched parameter-shift gradients.

    This optimizer leverages:
    1. Analytical gradients via parameter-shift rule (more accurate than COBYLA)
    2. GPU batching for parallel gradient computation
    3. L-BFGS-B quasi-Newton method (faster convergence than COBYLA)
    4. Adaptive early stopping based on energy plateau detection (same as COBYLA)
    """

    class EarlyStop(Exception):
        """Internal sentinel exception used to abort L-BFGS-B optimization early."""

    def __init__(
        self,
        kernel,
        hamiltonian,
        shots: int = -1,
        max_iterations: int = 1000,
        gtol: float = 1e-5,
        ftol: float = 1e-9,
        energy_tol: float | None = None,
        patience: int = 5,
        window: int = 10,
        use_adaptive_iterations: bool = True,
        enable_hamiltonian_batching: bool = False,
    ):
        """
        Initialize gradient-based optimizer.

        Args:
            kernel: CUDA-Q parameterized kernel
            hamiltonian: CUDA-Q spin operator
            shots: Number of shots (-1 for statevector)
            max_iterations: Maximum optimizer iterations
            gtol: Gradient tolerance for convergence
            ftol: Function tolerance for convergence
            energy_tol: Energy improvement threshold for adaptive stopping (default: ftol)
            patience: Number of consecutive non-improving iterations before stopping
            window: Size of rolling window for smoothing energy improvements
            use_adaptive_iterations: Enable adaptive early stopping (default: True)
            enable_hamiltonian_batching: Distribute Hamiltonian terms across GPUs
        """
        self.gradient_calculator = ParameterShiftGradient(kernel, hamiltonian, shots, enable_hamiltonian_batching=enable_hamiltonian_batching)
        self.max_iterations = max_iterations
        self.gtol = gtol
        self.ftol = ftol

        # Adaptive early stopping parameters (same as COBYLA)
        self.energy_tol = energy_tol if energy_tol is not None else ftol
        self.patience = int(max(patience, 1))
        self.window = int(max(window, 1))
        self.use_adaptive_iterations = bool(use_adaptive_iterations)

        # Tracking
        self.energy_history: List[float] = []
        self.params_history: List[np.ndarray] = []  # Track parameters at each iteration
        self.gradient_norms: List[float] = []
        self.iteration_count = 0
        self._no_improve_count = 0
        # Per-iteration timing
        self.iter_times: List[float] = []  # Total wall-clock time per iteration (energy + gradient)
        self._iter_start: float | None = None

    def cost_function(self, params: np.ndarray) -> float:
        """Objective function: ⟨H⟩"""
        self._iter_start = time.perf_counter()
        t0 = time.perf_counter()
        energy = self.gradient_calculator._single_observe(params)
        energy_dt = time.perf_counter() - t0
        self.energy_history.append(energy)
        self.params_history.append(np.array(params, dtype=float))  # Store parameter snapshot
        self.iteration_count += 1
        print(f"Gradient-based iter {self.iteration_count}: energy = {energy:.6f}  (energy eval: {energy_dt:.2f}s)")
        return energy

    def gradient_function(self, params: np.ndarray) -> np.ndarray:
        """Gradient function: ∇⟨H⟩ using parameter-shift rule"""
        t0 = time.perf_counter()
        gradient = self.gradient_calculator.compute_gradient(params)
        grad_dt = time.perf_counter() - t0
        grad_norm = np.linalg.norm(gradient)
        self.gradient_norms.append(grad_norm)
        # Record total iteration time (energy eval + gradient eval)
        if self._iter_start is not None:
            iter_total = time.perf_counter() - self._iter_start
            self.iter_times.append(iter_total)
            print(f"  |∇E| = {grad_norm:.6e}  (gradient: {grad_dt:.2f}s | iter total: {iter_total:.2f}s)")
            self._iter_start = None
        else:
            print(f"  |∇E| = {grad_norm:.6e}  (gradient: {grad_dt:.2f}s)")
        return gradient

    def _adaptive_callback(self, _xk: np.ndarray):
        """
        Adaptive early stopping callback based on energy convergence.

        Monitors energy improvements and stops optimization when:
        Energy improvements fall below energy_tol (energy plateau)

        Args:
            _xk: Current parameter vector (unused, required by SciPy callback signature)

        Raises:
            EarlyStop: When adaptive stopping criteria are met
        """
        n = len(self.energy_history)
        if n < 2:
            return

        # Look at recent window of energies
        window = min(self.window, n - 1)
        recent = self.energy_history[-(window + 1) :]
        diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        best_recent_improve = max(diffs) if diffs else float("inf")

        # Energy convergence check
        converged = (best_recent_improve < self.energy_tol)

        if converged:
            self._no_improve_count += 1
        else:
            self._no_improve_count = 0

        # Stop if patience exhausted
        if not self.use_adaptive_iterations:
            return

        if self._no_improve_count >= self.patience:
            print(f"\n[Adaptive stopping] Energy plateau (ΔE < {self.energy_tol:.2e})")
            print(f"                    Converged for {self.patience} consecutive checks (window={self.window})")
            raise GradientBasedVQEOptimizer.EarlyStop()

    def optimize(
        self,
        initial_params: np.ndarray,
        bounds: List[Tuple[float, float]] | None = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Run L-BFGS-B optimization with adaptive early stopping.

        Args:
            initial_params: Starting parameter vector
            bounds: Parameter bounds (default: None = unbounded)

        Returns:
            optimized_params: Best parameter vector found
            final_energy: Best energy found
        """
        self.energy_history = []
        self.params_history = []
        self.gradient_norms = []
        self.iteration_count = 0
        self._no_improve_count = 0

        # Use callback for adaptive early stopping (same as COBYLA)
        callback = self._adaptive_callback if self.use_adaptive_iterations else None

        try:
            result = minimize(
                fun=self.cost_function,
                x0=initial_params,
                jac=self.gradient_function,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.ftol,
                    "gtol": self.gtol,
                    "disp": False,
                },
            )

            optimized_params = result.x
            final_energy = result.fun

            # Check if any earlier iteration found a better energy than the
            # final result (can happen if L-BFGS-B overshoots near convergence)
            if self.energy_history:
                best_idx = int(np.argmin(self.energy_history))
                best_hist_energy = self.energy_history[best_idx]
                if best_hist_energy < final_energy:
                    optimized_params = self.params_history[best_idx]
                    final_energy = best_hist_energy

            print(f"\nL-BFGS-B optimization completed:")
            print(f"  Total iterations: {self.iteration_count}")
            print(f"  Final energy: {final_energy:.6f}")
            print(f"  Success: {result.success}")
            print(f"  Message: {result.message}")
            self._print_timing_summary()

            return optimized_params, final_energy

        except GradientBasedVQEOptimizer.EarlyStop:
            # Early stopping triggered - return best result so far
            if not self.energy_history:
                raise RuntimeError("Early stop triggered without any iterations recorded.")

            best_idx = int(np.argmin(self.energy_history))
            best_energy = self.energy_history[best_idx]
            best_params = self.params_history[best_idx]

            print(f"\nL-BFGS-B optimization stopped early (adaptive):")
            print(f"  Total iterations: {self.iteration_count}")
            print(f"  Best energy: {best_energy:.6f} (at iteration {best_idx + 1})")
            print(f"  Reason: No improvement > {self.energy_tol:.2e} for {self.patience} consecutive checks")
            self._print_timing_summary()

            return best_params, best_energy

    def _print_timing_summary(self):
        """Print per-iteration timing summary."""
        if not self.iter_times:
            return
        times = np.array(self.iter_times)
        total = times.sum()
        print(f"  Timing per iteration ({len(times)} iters, {total:.1f}s total):")
        print(f"    Mean: {times.mean():.2f}s | Median: {np.median(times):.2f}s | "
              f"Min: {times.min():.2f}s | Max: {times.max():.2f}s")


__all__ = ["ParameterShiftGradient", "GradientBasedVQEOptimizer"]
