from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union
import cudaq
from math import pi

# Reuse utilities and types from EfficientSU2 builder
from .efficientsu2 import pairs_for_pattern, apply_su2_gate, GateSpec


def improved_efficient_su2_cudaq(
    num_qubits: int,
    reps: int = 3,
    su2_gates: Sequence[GateSpec] = ('ry', 'rz'),
    entanglement: Union[str, List[Tuple[int,int]], Callable[[int,int], List[Tuple[int,int]]]] = 'reverse_linear',
    skip_unentangled_qubits: bool = False,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    initial_state: Union[None, str, Callable[[cudaq.Kernel, object], None], Callable[[object], None]] = None,
    parameter_prefix: str = 'θ',
    use_alternating_cx: bool = True,  # NEW: Enable alternating CX directions
):
    """
    Improved EfficientSU2 with alternating CX orientations (2-head pattern).

    This is a minimal improvement over standard EfficientSU2:
    - Same parameters (48 for 6 qubits, reps=3)
    - Same CX gate count
    - ONLY difference: Alternates CX(control, target) and CX(target, control)

    This mimics the "2-head" multi-head attention pattern but with clean EfficientSU2 code.

    Args:
        num_qubits: Number of qubits
        reps: Number of repetition layers
        su2_gates: Sequence of gates (e.g., 'ry', 'rz')
        entanglement: Entanglement pattern
        use_alternating_cx: If True, alternate CX directions (even pairs normal, odd pairs flipped)

    Returns:
        kernel, param_order, bounds
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be >= 1")

    # Determine rotation qubits per layer
    rotation_qubits_by_layer: List[List[int]] = []
    ent_pairs_by_layer: List[List[Tuple[int,int]]] = []

    for ell in range(reps):
        pairs = pairs_for_pattern(num_qubits, ell, entanglement)
        ent_pairs_by_layer.append(pairs)
        if skip_unentangled_qubits:
            touched = sorted({q for p in pairs for q in p})
            rotation_qubits_by_layer.append(touched)
        else:
            rotation_qubits_by_layer.append(list(range(num_qubits)))

    # Final rotation layer
    final_rots_qubits: List[int] = []
    if not skip_final_rotation_layer:
        if skip_unentangled_qubits and reps > 0:
            final_rots_qubits = rotation_qubits_by_layer[-1]
        else:
            final_rots_qubits = list(range(num_qubits))

    # Parameter layout (same as EfficientSU2)
    param_order: List[Tuple[str,int,int,int]] = []
    param_counter = 0

    for ell in range(reps):
        qubits = rotation_qubits_by_layer[ell]
        for q in qubits:
            for g_idx, g in enumerate(su2_gates):
                if isinstance(g, str) and g.lower() in ('x','y','z'):
                    continue
                gate_name = str(getattr(g, '__name__', g))
                param_name = f"{parameter_prefix}[{param_counter}]_{gate_name}_q{q}_L{ell}"
                param_order.append((param_name, q, ell, g_idx))
                param_counter += 1

    if final_rots_qubits:
        for q in final_rots_qubits:
            for g_idx, g in enumerate(su2_gates):
                if isinstance(g, str) and g.lower() in ('x','y','z'):
                    continue
                gate_name = str(getattr(g, '__name__', g))
                param_name = f"{parameter_prefix}[{param_counter}]_{gate_name}_q{q}_L{reps}"
                param_order.append((param_name, q, reps, g_idx))
                param_counter += 1

    num_params = len(param_order)
    bounds = [(-pi, pi) for _ in range(num_params)]

    # Build kernel
    mk = cudaq.make_kernel(*([float] * num_params))
    if isinstance(mk, tuple):
        kernel = mk[0]
        thetas = list(mk[1:])
    else:
        kernel = mk
        thetas = [kernel.argument(i) for i in range(num_params)]

    qvec = kernel.qalloc(num_qubits)

    # Initial state
    if initial_state is not None:
        if isinstance(initial_state, str):
            state = initial_state.lower().strip()
            if state == '+':
                for q in range(num_qubits):
                    kernel.h(qvec[q])
            elif state == '-':
                for q in range(num_qubits):
                    kernel.x(qvec[q])
                    kernel.h(qvec[q])
            elif state == '0':
                pass
            elif state == '1':
                for q in range(num_qubits):
                    kernel.x(qvec[q])
            else:
                raise ValueError(f"Unsupported initial_state: '{initial_state}'")
        elif callable(initial_state):
            try:
                initial_state(kernel, qvec)
            except TypeError:
                initial_state(qvec)
        else:
            raise ValueError("initial_state must be string or callable")

    p = 0  # Parameter cursor

    # Rotation layer helper
    def rotation_layer(qubits: Iterable[int]):
        nonlocal p
        for q in qubits:
            for g in su2_gates:
                if isinstance(g, str) and g.lower() in ('x','y','z'):
                    apply_su2_gate(kernel, g, 0.0, qvec[q])
                else:
                    apply_su2_gate(kernel, g, thetas[p], qvec[q])
                    p += 1

    # Build circuit: (rotations -> CX) × reps
    for ell in range(reps):
        if insert_barriers and ell > 0:
            pass

        rotation_layer(rotation_qubits_by_layer[ell])

        if insert_barriers:
            pass

        # NEW: Alternating CX pattern (2-head logic)
        for pair_idx, (c, t) in enumerate(ent_pairs_by_layer[ell]):
            if use_alternating_cx and (pair_idx % 2 == 1):
                # Odd-indexed pairs: flip orientation
                kernel.cx(qvec[t], qvec[c])
            else:
                # Even-indexed pairs: normal orientation
                kernel.cx(qvec[c], qvec[t])

    if final_rots_qubits:
        rotation_layer(final_rots_qubits)

    assert p == num_params, f"Parameter mismatch: {p} vs {num_params}"

    return kernel, param_order, bounds


