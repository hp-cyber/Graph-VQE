"""
Graph-ADAPT-VQE: Global ADAPT ansatz + Hamiltonian-graph block optimization.

Three-phase algorithm:
  1. Global ADAPT-VQE: Grow adaptive ansatz over ALL qubits (identical to
     AdaptVQE). Produces a kernel and parameter vector.
  2. Block optimization: Partition qubits via Hamiltonian interaction graph.
     For each block, run COBYLA on params whose ops act within that block,
     holding all other params fixed (coordinate descent).
  3. Global refinement: COBYLA on all params together.

The ansatz built in Phase 1 is EXACTLY the same as AdaptVQE — the only
difference is that Phases 2-3 further refine the parameters using
graph-aware coordinate descent instead of stopping after Phase 1.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cudaq
import numpy as np
from scipy.optimize import minimize

from .subspace_vqe import SubspaceVQE
from .adapt_vqe import AdaptVQE, Op


class GraphAdaptVQE(SubspaceVQE):
    """
    Graph-ADAPT-VQE.

    Inherits Hamiltonian-aware graph partitioning from SubspaceVQE.
    Overrides run_vqe() to:
      1. Build the ADAPT ansatz globally (same as AdaptVQE).
      2. Refine parameters block-by-block on the ADAPT kernel.
      3. Apply global COBYLA refinement.
    """

    def __init__(
        self,
        hamiltonian,
        num_qubits: int,
        entanglement: str = "circular",
        reps_for_pool: int = 4,
        gradient_threshold: float = 1e-3,
        eigenvalue_threshold: float = 1e-5,
        adapt_max_steps: Optional[int] = None,
        cobyla_maxiter: int = 1000000,
        **subspace_kwargs,
    ):
        super().__init__(
            hamiltonian=hamiltonian,
            num_qubits=num_qubits,
            entanglement=entanglement,
            **subspace_kwargs,
        )
        self.entanglement = entanglement
        self.reps_for_pool = int(reps_for_pool)
        self.gradient_threshold = float(gradient_threshold)
        self.eigenvalue_threshold = float(eigenvalue_threshold)
        self.adapt_max_steps = adapt_max_steps
        self.cobyla_maxiter = int(cobyla_maxiter)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adapt_observe(self, kernel, params: np.ndarray) -> float:
        """Evaluate full H expectation on the ADAPT kernel."""
        self.nfev += 1
        args = params.tolist()
        if self.shots == -1:
            result = cudaq.observe(kernel, self.hamiltonian, *args)
        else:
            result = cudaq.observe(kernel, self.hamiltonian, *args,
                                   shots_count=self.shots)
        if hasattr(result, "expectation_value"):
            return float(result.expectation_value)
        exp = result.expectation
        return float(exp() if callable(exp) else exp)

    def _block_param_indices(self, selected_ops: List[Op],
                              block_qubits: List[int]) -> List[int]:
        """Indices of ADAPT params whose op acts entirely within block_qubits."""
        block_set = set(block_qubits)
        return [
            i for i, (gate, q0, q1) in enumerate(selected_ops)
            if q0 in block_set and (q1 is None or q1 in block_set)
        ]

    def _cobyla_subset(
        self,
        kernel,
        params_full: np.ndarray,
        subset_idx: List[int],
        maxiter: int,
        patience: int,
    ) -> Tuple[np.ndarray, float]:
        """COBYLA on a subset of params, holding the rest fixed."""
        energy_hist: List[float] = []
        params_hist: List[np.ndarray] = []
        no_improve = [0]

        def cost(xk):
            p = params_full.copy()
            p[subset_idx] = xk
            e = self._adapt_observe(kernel, p)
            energy_hist.append(e)
            params_hist.append(np.array(xk, copy=True))
            return e

        class _ES(Exception):
            pass

        def cb(_xk):
            n = len(energy_hist)
            if n < 2:
                return
            w = min(self.window, n - 1)
            recent = energy_hist[-(w + 1):]
            best_imp = max(abs(recent[i] - recent[i - 1])
                           for i in range(1, len(recent)))
            if best_imp < self.energy_tol:
                no_improve[0] += 1
            else:
                no_improve[0] = 0
            if no_improve[0] >= patience:
                raise _ES()

        try:
            res = minimize(cost, params_full[subset_idx], method="cobyla",
                           options={"maxiter": maxiter}, callback=cb)
            x_opt = np.array(res.x, dtype=float)
            e_opt = float(res.fun)
        except _ES:
            if energy_hist:
                bi = int(np.argmin(energy_hist))
                x_opt = params_hist[bi]
                e_opt = energy_hist[bi]
            else:
                x_opt = params_full[subset_idx].copy()
                e_opt = float("inf")

        self.energy_history.extend(energy_hist)
        result = params_full.copy()
        result[subset_idx] = x_opt
        return result, e_opt

    def _cobyla_global(
        self,
        kernel,
        params: np.ndarray,
        maxiter: int,
        patience: int,
    ) -> Tuple[np.ndarray, float]:
        """COBYLA on all params."""
        energy_hist: List[float] = []
        params_hist: List[np.ndarray] = []
        no_improve = [0]

        def cost(xk):
            e = self._adapt_observe(kernel, np.array(xk, dtype=float))
            energy_hist.append(e)
            params_hist.append(np.array(xk, copy=True))
            return e

        class _ES(Exception):
            pass

        def cb(_xk):
            n = len(energy_hist)
            if n < 2:
                return
            w = min(self.window, n - 1)
            recent = energy_hist[-(w + 1):]
            best_imp = max(abs(recent[i] - recent[i - 1])
                           for i in range(1, len(recent)))
            if best_imp < self.energy_tol:
                no_improve[0] += 1
            else:
                no_improve[0] = 0
            if no_improve[0] >= patience:
                raise _ES()

        try:
            res = minimize(cost, params, method="cobyla",
                           options={"maxiter": maxiter}, callback=cb)
            x_opt = np.array(res.x, dtype=float)
            e_opt = float(res.fun)
        except _ES:
            if energy_hist:
                bi = int(np.argmin(energy_hist))
                x_opt = params_hist[bi]
                e_opt = energy_hist[bi]
            else:
                x_opt = params.copy()
                e_opt = float("inf")

        self.energy_history.extend(energy_hist)
        return x_opt, e_opt

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_vqe(self):
        """
        Run Graph-ADAPT-VQE.

        Phase 1: Global ADAPT — builds the same adaptive ansatz as AdaptVQE.
        Phase 2: Block optimization — graph-partitioned COBYLA coordinate descent.
        Phase 3: Global refinement — full-parameter COBYLA polish.
        """
        self.energy_history = []
        self.nfev = 0

        # ----------------------------------------------------------
        # Phase 1: Global ADAPT (identical ansatz to AdaptVQE)
        # ----------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"Graph-ADAPT-VQE  Phase 1: Global ADAPT ansatz building")
        print(f"{'='*60}")

        adapt = AdaptVQE(
            hamiltonian=self.hamiltonian,
            num_qubits=self.num_qubits,
            entanglement=self.entanglement,
            reps_for_pool=self.reps_for_pool,
            gradient_threshold=self.gradient_threshold,
            eigenvalue_threshold=self.eigenvalue_threshold,
            max_iterations=self.adapt_max_steps,
            max_iter=self.cobyla_maxiter,
            energy_tol=self.energy_tol,
            patience=self.patience,
            window=self.window,
            shots=self.shots,
            seed=self.seed,
        )
        _, adapt_params, kernel, _ = adapt.run_vqe()
        selected_ops = list(adapt.selected_ops)
        self.energy_history.extend(adapt.energy_history)
        self.nfev += adapt.nfev

        if not selected_ops:
            print("ADAPT produced no operators — returning initial state.")
            return self.energy_history, adapt_params, kernel, \
                   [(float("inf"), adapt_params)]

        best_params = adapt_params.copy()
        best_energy = min(adapt.energy_history) if adapt.energy_history \
            else float("inf")
        print(f"Phase 1 complete. Ansatz: {len(selected_ops)} ops, "
              f"best energy: {best_energy:.6f}")

        # ----------------------------------------------------------
        # Phase 2: Block optimization on ADAPT params
        # ----------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"Graph-ADAPT-VQE  Phase 2: Block optimization")
        print(f"{'='*60}")

        blocks = self._partition_hamiltonian_graph()
        print(f"Hamiltonian-aware blocks: {blocks}")

        epoch_no_improve = 0
        epoch_best = best_energy

        for epoch in range(self.max_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.max_epochs} ---")
            epoch_params = best_params.copy()

            for i, block_qubits in enumerate(blocks):
                idx = self._block_param_indices(selected_ops, block_qubits)
                if not idx:
                    print(f"  Block {i+1} {block_qubits}: "
                          f"no in-block ADAPT params, skip")
                    continue
                print(f"  Block {i+1} {block_qubits}: "
                      f"optimising {len(idx)} params")
                epoch_params, blk_e = self._cobyla_subset(
                    kernel, epoch_params, idx,
                    maxiter=self.block_maxiter,
                    patience=self.block_patience,
                )
                print(f"  Block {i+1} energy: {blk_e:.6f}")

            # Global refinement after each epoch
            if self.global_refinement:
                print(f"  Global refinement...")
                epoch_params, epoch_e = self._cobyla_global(
                    kernel, epoch_params,
                    maxiter=self.global_maxiter,
                    patience=self.global_patience,
                )
                print(f"  Global energy: {epoch_e:.6f}")
            else:
                epoch_e = self._adapt_observe(kernel, epoch_params)
                self.energy_history.append(epoch_e)

            if epoch_e < best_energy:
                best_energy = epoch_e
                best_params = epoch_params.copy()

            # Epoch-level early stopping
            if epoch_e < epoch_best - self.energy_tol:
                epoch_best = epoch_e
                epoch_no_improve = 0
            else:
                epoch_no_improve += 1
                if epoch_no_improve >= self.epoch_patience:
                    print(f"Epoch early stop: no improvement for "
                          f"{self.epoch_patience} epochs")
                    break

        print(f"\n{'='*60}")
        print(f"Graph-ADAPT-VQE complete. Best energy: {best_energy:.6f}")
        print(f"{'='*60}\n")

        top_results = [(best_energy, best_params)]
        return self.energy_history, best_params, kernel, top_results
