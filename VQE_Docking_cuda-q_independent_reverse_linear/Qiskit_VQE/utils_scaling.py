"""
Utilities for EfficientNet-style compound scaling of VQE hyperparameters.

The core helper `compound_scale` maps a scalar stage index ``phi`` to a tuple of
ansatz depth (`reps`), entanglement topology, whether to skip rotations on
unentangled qubits, and measurement shots. The defaults follow the guidance in
EfficientNet.md and can be overridden as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

DEFAULT_ENTANGLEMENT_SCHEDULE: Tuple[str, ...] = ("pairwise", "linear", "circular", "full")


@dataclass(frozen=True)
class CompoundScaleConfig:
    """Container for compound scaling hyperparameters."""

    d0: int = 2
    r0: int = 128
    alpha: float = 1.2
    gamma: float = 2.0
    entanglement_schedule: Sequence[str] = DEFAULT_ENTANGLEMENT_SCHEDULE
    skip_unentangled_phi_threshold: float = 1.0
    statevector_phis: Iterable[int] | None = None
    min_shots: int = 1


def compound_scale(
    phi: float,
    *,
    d0: int = 2,
    r0: int = 128,
    alpha: float = 1.2,
    gamma: float = 2.0,
    entanglement_schedule: Sequence[str] | None = None,
    skip_unentangled_phi_threshold: float = 1.0,
    statevector_phis: Iterable[int] | None = None,
    min_shots: int = 1,
) -> Tuple[int, str, bool, int]:
    """
    Map a scale index ``phi`` to (reps, entanglement, skip_unentangled_qubits, shots).

    Parameters
    ----------
    phi
        Scale index (integer or float). Higher values increase depth, connectivity,
        and measurement budget.
    d0, alpha
        Base depth and multiplicative factor for ``reps``. Depth is rounded to the
        nearest integer and clipped to be at least 1.
    r0, gamma
        Base measurement shots and multiplicative factor. The value is rounded to
        the nearest integer and clipped to be at least ``min_shots`` unless ``phi``
        is marked as statevector (shots = -1).
    entanglement_schedule
        Ordered sequence of entanglement patterns. ``phi`` chooses the element at
        position ``min(int(phi), len(schedule) - 1)``.
    skip_unentangled_phi_threshold
        If ``phi`` is less or equal to this value, only qubits that participate in
        entanglement will receive parameterised rotations (i.e. "narrow" width).
    statevector_phis
        Iterable of phi indices where we should use statevector simulation
        (shots = -1). Useful for warm-up stages.
    min_shots
        Lower bound for shot-based stages (ignored when statevector is requested).

    Returns
    -------
    Tuple[int, str, bool, int]
        (reps, entanglement pattern, skip_unentangled_qubits, shots)
    """

    if entanglement_schedule is None:
        entanglement_schedule = DEFAULT_ENTANGLEMENT_SCHEDULE

    reps = max(1, round(d0 * (alpha ** phi)))

    idx = int(phi)
    if idx < 0:
        idx = 0
    if idx >= len(entanglement_schedule):
        idx = len(entanglement_schedule) - 1
    entanglement = entanglement_schedule[idx]

    skip_unentangled = phi <= skip_unentangled_phi_threshold

    if statevector_phis and int(phi) in set(int(v) for v in statevector_phis):
        shots = -1
    else:
        computed_shots = round(r0 * (gamma ** phi))
        if computed_shots < min_shots:
            computed_shots = min_shots
        shots = int(computed_shots)

    return reps, entanglement, skip_unentangled, shots


__all__ = ["CompoundScaleConfig", "DEFAULT_ENTANGLEMENT_SCHEDULE", "compound_scale"]
