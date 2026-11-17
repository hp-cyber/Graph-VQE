"""
CVaR-VQE: Conditional Value at Risk Variational Quantum Eigensolver.

Replaces the standard expectation value E[H] with CVaR_alpha[H], defined as
the average energy of the alpha-best (lowest) measurement outcomes.

Reference: Barkoutsos et al. (2020) "Improving Variational Quantum Optimization
using CVaR" — Quantum 4, 256. Alpha=0.25 recommended (tutorial default).

Implementation follows qiskit-optimization tutorial 08_cvar_optimization:
  SamplingVQE(..., aggregation=alpha)
adapted for CUDA-Q: uses cudaq.sample() and evaluates diagonal Hamiltonian
terms (Z/I Paulis only) per bitstring — same physics as QUBO-style CVaR.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Tuple

import cudaq
import numpy as np
from scipy.optimize import minimize

from .cudaq import CUDAQVQE, qiskit_sparsepauliop_to_cudaq_spin
from .efficientsu2 import efficient_su2_cudaq


class CVaRVQE(CUDAQVQE):
    """
    CVaR-VQE: same EfficientSU2 ansatz as baseline but optimises the
    alpha-CVaR of bitstring energies instead of the full expectation value.

    Energy per bitstring is computed from Z/I Pauli terms only (the diagonal
    part of H in the computational basis).  X/Y terms contribute 0 to a
    computational-basis eigenstate, so this is exact for diagonal Hamiltonians
    and a variational lower bound otherwise.

    alpha=0.25 (tutorial default) focuses on the best 25% of samples, which
    improves convergence on combinatorial energy landscapes.
    """

    def __init__(
        self,
        hamiltonian,
        alpha: float = 0.25,
        cvar_shots: int = 10000,
        **kwargs,
    ):
        """
        Args:
            hamiltonian: CUDA-Q SpinOperator.
            alpha: CVaR confidence level in (0, 1].  alpha=1 → standard
                   expectation value; alpha→0 → minimum-energy shot only.
                   Recommended: 0.25 (Barkoutsos et al. 2020).
            cvar_shots: Shots for cudaq.sample() per energy evaluation.
                        Ignored for analytic terms; used for CVaR computation.
            **kwargs: Forwarded to CUDAQVQE (max_iterations, shots, reps,
                      entanglement, energy_tol, patience, window, seed, …).
        """
        # CVaR always needs shot-based sampling; force shots > 0 for the sampler
        kwargs.setdefault("shots", cvar_shots)
        super().__init__(hamiltonian=hamiltonian, **kwargs)
        self.alpha = float(alpha)
        self.cvar_shots = int(cvar_shots)

        # Pre-parse diagonal (Z/I-only) Hamiltonian terms for bitstring scoring
        self._diag_terms: List[Tuple[float, List[int]]] = []
        self._parse_diagonal_terms()

    # ------------------------------------------------------------------
    # Diagonal Hamiltonian parsing
    # ------------------------------------------------------------------

    def _parse_diagonal_terms(self):
        """
        Extract terms with only Z and I Paulis (diagonal in Z-basis).
        Stores list of (coeff, z_qubit_indices) for fast per-bitstring eval.
        """
        self._diag_terms = []
        try:
            for term in self.hamiltonian:
                pauli_str = str(term).strip()
                if "*" not in pauli_str:
                    continue
                coeff_part, pauli_part = pauli_str.split("*", 1)
                coeff_part = coeff_part.strip().replace("(", "").replace(")", "")
                # Parse complex coefficient, keep real part
                try:
                    coeff_part_j = coeff_part.replace("i", "j")
                    coeff = complex(coeff_part_j).real
                except ValueError:
                    coeff = float(coeff_part.split("+")[0].split("-")[0])
                pauli = pauli_part.strip()

                # Only keep terms that are all Z or I
                if all(c in "ZI" for c in pauli):
                    z_indices = [i for i, c in enumerate(pauli) if c == "Z"]
                    self._diag_terms.append((coeff, z_indices))
        except Exception:
            # Fallback: no diagonal terms parsed; CVaR will fall back to observe
            self._diag_terms = []

    def _bitstring_energy(self, bitstring: str) -> float:
        """
        Diagonal energy of a single measurement outcome bitstring.
        CUDA-Q uses big-endian convention: bitstring[0] = qubit 0.
        Z eigenvalue on qubit i for bit b_i: +1 if b_i='0', -1 if b_i='1'.
        """
        energy = 0.0
        for coeff, z_qubits in self._diag_terms:
            sign = 1.0
            for qi in z_qubits:
                if qi < len(bitstring) and bitstring[qi] == "1":
                    sign *= -1.0
            energy += coeff * sign
        return energy

    # ------------------------------------------------------------------
    # CVaR energy evaluation (replaces cudaq.observe)
    # ------------------------------------------------------------------

    def _cvar_energy(self, params: Iterable[float]) -> float:
        """
        Evaluate CVaR_alpha energy:
          1. Sample the circuit with cvar_shots shots.
          2. Score each unique bitstring by its diagonal Hamiltonian energy.
          3. Expand by counts, sort ascending, take lowest alpha fraction.
          4. Return their average.
        Falls back to cudaq.observe if no diagonal terms were parsed.
        """
        if not self._diag_terms:
            # No diagonal terms — fall back to full expectation value
            return self.observe_energy(params)

        params_list = list(params)
        try:
            counts = cudaq.sample(self.kernel, *params_list, shots_count=self.cvar_shots)
        except Exception:
            return self.observe_energy(params)

        # Build flat list of (energy, count) pairs
        scored: List[Tuple[float, int]] = []
        try:
            for bitstring, count in counts.items():
                e = self._bitstring_energy(bitstring)
                scored.append((e, int(count)))
        except Exception:
            return self.observe_energy(params)

        if not scored:
            return self.observe_energy(params)

        # Expand, sort ascending (lowest energy = best)
        flat = []
        for e, cnt in scored:
            flat.extend([e] * cnt)
        flat.sort()

        # CVaR: average of lowest alpha fraction
        k = max(1, int(np.ceil(self.alpha * len(flat))))
        cvar = float(np.mean(flat[:k]))
        return cvar

    # ------------------------------------------------------------------
    # Override cost_func used by COBYLA in the parent class
    # ------------------------------------------------------------------

    def cost_func(self, params: np.ndarray) -> float:
        """Override parent's cost_func to use CVaR instead of observe."""
        params_list = list(params)
        energy = self._cvar_energy(params_list)
        self.energy_list.append(energy)
        self._stage_energy.append(energy)
        self._stage_iteration_results.append((energy, np.array(params_list)))
        self._stage_callback(params)
        return energy
