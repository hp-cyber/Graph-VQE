import numpy as np
import cudaq
from scipy.optimize import minimize
import itertools
from typing import Callable, Iterable, Sequence
from qiskit.quantum_info import SparsePauliOp
from cudaq import spin
from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union


# Change GateSpec so callables get (kernel, theta, q)
GateSpec = Union[str, Callable[[cudaq.Kernel, float, object], None]]

def apply_su2_gate(kernel: cudaq.Kernel, name_or_fn: GateSpec, theta: float, q):
    if callable(name_or_fn):
        name_or_fn(kernel, theta, q)  # user-defined hook
        return
    g = name_or_fn.lower()
    if g == 'rx': kernel.rx(theta, q)
    elif g == 'ry': kernel.ry(theta, q)
    elif g == 'rz': kernel.rz(theta, q)
    elif g == 'x':  kernel.x(q)            # non-parametric
    elif g == 'y':  kernel.y(q)
    elif g == 'z':  kernel.z(q)
    else:
        raise ValueError(f"Unsupported SU(2) gate spec: {name_or_fn}")

def pairs_for_pattern(n: int, layer: int, entanglement: Union[str, List[Tuple[int,int]], Callable[[int,int], List[Tuple[int,int]]]]) -> List[Tuple[int,int]]:
    if callable(entanglement):
        try:
            pairs = entanglement(layer)                # Qiskit-style
        except TypeError:
            pairs = entanglement(n, layer)             # your extended style
        return [tuple(p) for p in pairs]

    if isinstance(entanglement, list):
        return [tuple(p) for p in entanglement]

    pat = entanglement.lower()
    if pat == 'full':
        return [(i, j) for i in range(n) for j in range(i+1, n)]
    if pat == 'linear':
        return [(i, i+1) for i in range(n-1)]
    
    # CORRECTED: reverse_linear is static - no layer-dependent rotation
    if pat == 'reverse_linear':
        # Reverse the order of linear pairs
        return [(i, i+1) for i in range(n-2, -1, -1)]
    
    if pat == 'circular':
        # Circular is linear + wrap-around
        return [(i, i+1) for i in range(n-1)] + [(n-1, 0)]
    
    if pat == 'pairwise':
        # alternate even/odd pairings each layer
        base = 0 if layer % 2 == 0 else 1
        return [(i, i+1) for i in range(base, n-1, 2)]
    
    # sca (shifted-circular-alternating)
    if pat == 'sca':
        offset_idx = layer % n
        ret = []
        
        if offset_idx % 2 == 0:  # even layers
            # Build pairs going backward, then add wrap-around, then continue
            for i in reversed(range(offset_idx)):
                ret.append((i, i + 1))
            ret.append((n - 1, 0))  # the "long" entanglement (shifted position)
            for i in reversed(range(offset_idx + 1, n)):
                ret.append((i - 1, i))
        else:  # odd layers - swap control/target
            for i in range(n - offset_idx - 1, n - 1):
                ret.append((i + 1, i))  # swapped!
            ret.append((0, n - 1))  # the "long" entanglement (shifted, swapped)
            for i in range(n - offset_idx - 1):
                ret.append((i + 1, i))  # swapped!
        
        return ret
    
    raise ValueError(f"Unsupported entanglement pattern: {entanglement}")

def efficient_su2_cudaq(
    num_qubits: int,
    reps: int = 3,
    su2_gates: Sequence[GateSpec] = ('ry', 'rz'),
    entanglement: Union[str, List[Tuple[int,int]], Callable[[int,int], List[Tuple[int,int]]]] = 'reverse_linear',
    skip_unentangled_qubits: bool = False,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    initial_state: Union[None, cudaq.Kernel] = None,
    parameter_prefix: str = 'θ',
):
    """
    Build a CUDA-Q parameterized kernel for a hardware-efficient SU(2) ansatz.

    Args:
        num_qubits: Number of qubits in the circuit
        reps: Number of repetition layers
        su2_gates: Sequence of SU(2) gates to apply (e.g., 'ry', 'rz')
        entanglement: Entanglement pattern ('full', 'linear', 'reverse_linear', 'circular', 'pairwise', 'sca')
        skip_unentangled_qubits: If True, only apply rotations to entangled qubits
        skip_final_rotation_layer: If True, skip the final rotation layer
        insert_barriers: If True, insert barriers between layers (for visualization/debugging)
        initial_state: Optional initial state preparation. Can be:
            - String: '+' (all |+⟩), '-' (all |-⟩), '0' (all |0⟩), '1' (all |1⟩)
            - Callable: lambda k, q: ... or lambda q: ... to apply custom gates
            - None: defaults to |0...0⟩
        parameter_prefix: Prefix for parameter names in param_order metadata

    Returns:
        kernel: a cudaq.Kernel with len(params) float arguments
        param_order: List[(param_name, qubit_index, layer_index, gate_index_in_layer)]
                     describing the order of parameters expected by the kernel
        bounds: List[(low, high)] bounds per parameter, default (-pi, pi)
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be >= 1")

    # Determine, per layer, which qubits receive rotations (respect skip_unentangled_qubits)
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

    # Decide if there's a final rotation layer
    final_rots_qubits: List[int] = []
    if not skip_final_rotation_layer:
        if skip_unentangled_qubits and reps > 0:
            final_rots_qubits = rotation_qubits_by_layer[-1]
        else:
            # For reps==0 or when not skipping, act on all qubits.
            final_rots_qubits = list(range(num_qubits))

    # Parameter layout & count
    param_order: List[Tuple[str,int,int,int]] = []  # (param_name, qubit, layer, gate_idx)
    param_counter = 0
    # reps rotation+entanglement layers
    for ell in range(reps):
        qubits = rotation_qubits_by_layer[ell]
        for q in qubits:
            for g_idx, g in enumerate(su2_gates):
                if isinstance(g, str) and g.lower() in ('x','y','z'):  # non-param gates -> no parameter
                    continue
                gate_name = str(getattr(g, '__name__', g))
                param_name = f"{parameter_prefix}[{param_counter}]_{gate_name}_q{q}_L{ell}"
                param_order.append((param_name, q, ell, g_idx))
                param_counter += 1
    # final rotation layer (if any)
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

    # Build a kernel with num_params float arguments (supports old/new CUDA-Q APIs)
    mk = cudaq.make_kernel(*([float] * num_params))
    if isinstance(mk, tuple):
        # Newer CUDA-Q: (kernel, theta0, theta1, …)
        kernel = mk[0]
        thetas = list(mk[1:])
    else:
        # Older CUDA-Q: returns Kernel; fetch arguments from the Kernel
        kernel = mk
        thetas = [kernel.argument(i) for i in range(num_params)]

    qvec = kernel.qalloc(num_qubits)

    # Apply initial state preparation if provided
    if initial_state is not None:
        # initial_state can be:
        # 1. A string: '+' (all |+⟩), '-' (all |-⟩), '0' (all |0⟩), '1' (all |1⟩)
        # 2. A callable that takes (kernel, qvec) and applies gates
        # 3. A callable that takes just (qvec) and applies gates using the outer kernel

        if isinstance(initial_state, str):
            # Handle common named states
            state = initial_state.lower().strip()
            if state == '+':
                # |+⟩ = H|0⟩ on all qubits
                for q in range(num_qubits):
                    kernel.h(qvec[q])
            elif state == '-':
                # |-⟩ = X·H|0⟩ = H·Z|0⟩ on all qubits
                for q in range(num_qubits):
                    kernel.x(qvec[q])
                    kernel.h(qvec[q])
            elif state == '0':
                # |0⟩ on all qubits (default state, no-op)
                pass
            elif state == '1':
                # |1⟩ = X|0⟩ on all qubits
                for q in range(num_qubits):
                    kernel.x(qvec[q])
            else:
                raise ValueError(
                    f"Unsupported initial_state string: '{initial_state}'. "
                    "Supported strings: '+', '-', '0', '1'"
                )
        elif callable(initial_state):
            try:
                # Try calling with both kernel and qvec
                initial_state(kernel, qvec)
            except TypeError:
                try:
                    # Try calling with just qvec (assumes it closes over the kernel)
                    initial_state(qvec)
                except Exception as e:
                    raise ValueError(
                        f"initial_state callable failed: {e}. "
                        "It should accept either (kernel, qvec) or just (qvec). "
                        "Example: lambda k, q: k.x(q[0])"
                    )
        else:
            raise ValueError(
                "initial_state must be a string ('+', '-', '0', '1') or a callable. "
                "Example: '+' or lambda k, q: k.x(q[0])"
            )

    p = 0  # parameter cursor

    # Helper to apply one full "rotation layer"
    def rotation_layer(qubits: Iterable[int], layer_index: int):
        nonlocal p
        for q in qubits:
            for g in su2_gates:
                if isinstance(g, str) and g.lower() in ('x','y','z'):
                    apply_su2_gate(kernel, g, 0.0, qvec[q])
                else:
                    apply_su2_gate(kernel, g, thetas[p], qvec[q])
                    p += 1

    # Build the ansatz: (rot -> CX) * reps, then optional final rot
    for ell in range(reps):
        if insert_barriers and ell > 0:
            # Insert barrier between layers (CUDA-Q doesn't have native barriers,
            # but we can document this for future visualization support)
            # In practice, this is a no-op but preserved for API compatibility
            pass

        rotation_layer(rotation_qubits_by_layer[ell], ell)

        if insert_barriers:
            # Barrier after rotation layer, before entanglement
            pass

        # entanglement layer (CX)
        for c, t in ent_pairs_by_layer[ell]:
            kernel.cx(qvec[c], qvec[t]) 

    if final_rots_qubits:
        rotation_layer(final_rots_qubits, reps)

    # Sanity check
    assert p == num_params, f"Internal parameter count mismatch: used {p} vs expected {num_params}"

    return kernel, param_order, bounds
    

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
            if ch == 'I':
                continue
            elif ch == 'X':
                term = term * spin.x(q)
            elif ch == 'Y':
                term = term * spin.y(q)
            elif ch == 'Z':
                term = term * spin.z(q)
        H = H + (float(coeff.real) * term)
    return H

class CUDAQVQE:

    def __init__(self, hamiltonian, optimization_level=3, shots=200, max_iterations=20, seed=None):
        """
        Initialize CUDA-Q VQE solver.

        Args:
            hamiltonian: CUDA-Q spin operator
            optimization_level: Optimization level (unused, kept for compatibility)
            shots: Number of measurement shots. Use -1 for noiseless statevector simulation
            max_iterations: Maximum number of optimization iterations
            seed: Random seed for reproducibility (default: 42)
        """
        self.shots = shots
        self.hamiltonian = hamiltonian
        self.cost_history_dict = {"prev_vector": None, "iters": 0, "cost_history": []}
        self.iteration_results = []
        self.energy_list = []
        self.max_iterations = max_iterations
        self.optimization_level = optimization_level
        self.seed = seed if seed is not None else 42  # Default seed for reproducibility
        self.num_qubits: int | None = None
        if self.num_qubits is None:
            if hasattr(hamiltonian, "num_qubits"):
                self.num_qubits = hamiltonian.num_qubits
            elif hasattr(hamiltonian, "qubit_count"):
                qc_attr = getattr(hamiltonian, "qubit_count")
                self.num_qubits = qc_attr() if callable(qc_attr) else qc_attr
            else:
                raise ValueError("Please provide num_qubits explicitly (cannot infer from Hamiltonian).")

        self.kernel, param_order, self.param_bounds = efficient_su2_cudaq(num_qubits=self.num_qubits)
        self.kernel_param_count = len(param_order)

    def observe_energy(self, params: list[float]) -> float:
        """
        Evaluate expectation value <H> for given parameter vector.
        Handles CUDA-Q observe result across versions.

        For shots=-1, performs exact statevector simulation (noiseless).
        For shots>0, performs sampling-based measurement (with shot noise).
        """
        try:
            # Newer CUDA-Q: expand each parameter
            if self.shots == -1:
                # Noiseless simulation: no shots_count parameter
                obs_result = cudaq.observe(self.kernel, self.hamiltonian, *list(params))
            else:
                # Sampling-based simulation
                obs_result = cudaq.observe(self.kernel, self.hamiltonian, *list(params), shots_count=self.shots)
        except TypeError:
            # Older CUDA-Q: some builds accept the whole list as a single argument
            if self.shots == -1:
                obs_result = cudaq.observe(self.kernel, self.hamiltonian, list(params))
            else:
                obs_result = cudaq.observe(self.kernel, self.hamiltonian, list(params), shots_count=self.shots)

        # Robust extraction of energy
        if hasattr(obs_result, "expectation_value"):
            return float(getattr(obs_result, "expectation_value"))
        if hasattr(obs_result, "expectation"):
            exp_attr = getattr(obs_result, "expectation")
            return float(exp_attr() if callable(exp_attr) else exp_attr)
        if hasattr(obs_result, "energy"):
            return float(getattr(obs_result, "energy"))
        try:
            return float(obs_result)
        except Exception as exc:
            raise TypeError("Unsupported CUDA-Q observe result; cannot extract energy") from exc

    def cost_func(self, params: np.ndarray) -> float:
        energy = self.observe_energy(params)
        self.energy_list.append(energy)
        self.iteration_results.append((energy, np.array(params, dtype=float)))
        self.cost_history_dict["iters"] += 1
        self.cost_history_dict["prev_vector"] = np.array(params, dtype=float)
        self.cost_history_dict["cost_history"].append(energy)
        print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {energy}]")
        return energy

    def run_vqe(self):
        """
        Executes VQE using a SciPy COBYLA loop with per-iteration logging.

        Returns:
            energy_list: list of observed energies over all iterations
            best_params: optimized parameter vector
            kernel: the CUDA-Q ansatz kernel
            top_k_results: sorted list of (energy, params) across all iterations (top 6)
        """
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        x0 = np.random.random(self.kernel_param_count)
        # For COBYLA, use the user-specified max_iterations
        # Ensure it's at least num_params + 2 to avoid trivial optimization
        min_required = self.kernel_param_count + 2
        maxiter = max(min_required, self.max_iterations)
        res = minimize(self.cost_func, x0, method="cobyla", options={"maxiter": maxiter})

        sorted_results = sorted(self.iteration_results, key=lambda x: x[0])
        top_6_results = sorted_results[:6]

        return self.energy_list, np.array(res.x, dtype=float), self.kernel, top_6_results



    # def run_vqe(self):
    #     """
    #     cudaq.vqe(kernel: object, spin_operator: object, optimizer: optimizers.optimizer, parameter_count: int, shots: int = -1)→ tuple[float, list[float]]
    #     """
    #     opt = cudaq.optimizers.COBYLA()
    #     opt.max_iterations = self.max_iterations
    #     final_energy, best_params = cudaq.vqe(
    #         kernel=self.kernel,
    #         spin_operator=self.hamiltonian,
    #         optimizer=opt,
    #         parameter_count=self.kernel_param_count,
    #         shots=self.shots
    #     )
    #     # Bookkeeping to mirror your pattern
    #     self.energy_list.append(final_energy)
    #     self.iteration_results.append((final_energy, list(best_params)))
    #     self.cost_history_dict["iters"] += 1
    #     self.cost_history_dict["prev_vector"] = list(best_params)
    #     self.cost_history_dict["cost_history"].append(final_energy)
    #     print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {final_energy}]")

    #     # "Top results" (only one here, but kept for API symmetry)
    #     sorted_results = sorted(self.iteration_results, key=lambda x: x[0])
    #     top_6_results = sorted_results[:6]

    #     return self.energy_list, np.array(best_params, dtype=float), self.kernel, top_6_results