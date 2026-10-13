"""Hamiltonian graph communities, independent local VQEs, then full-H refinement.

The partition is a warm-start heuristic whenever Hamiltonian terms cross blocks.
No crossing term is discarded from the global objective. The global ansatz is
a single nearest-neighbor chain; local rotation parameters initialize it.
"""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import time

import cudaq
import networkx as nx
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from .cudaq import qiskit_sparsepauliop_to_cudaq_spin
from .efficientsu2 import pairs_for_pattern
from .mps_config import configure_full_bond, configure_gpu_groups
from .subspace_vqe import SubspaceVQE


class IndependentSubspaceVQE(SubspaceVQE):
    """Optimize each local circuit once, then run one joint optimization.

    max_block_qubits limits local workload, not the total problem size. Set it
    to None to keep entire graph communities. Singleton blocks use one idle
    spectator because the custom MPS backend requires two physical qubits.
    """

    def __init__(self, *, qiskit_hamiltonian, max_block_qubits=12, blocks=None,
                 energy_tol=1.0, shots=1000, **kwargs):
        kwargs["immediate_early_stopping"] = True
        kwargs["use_hamiltonian_blocking"] = True
        kwargs.setdefault("use_gradient_optimizer", False)
        kwargs.setdefault("entanglement", "linear")
        super().__init__(energy_tol=energy_tol, shots=shots, **kwargs)
        self.window_size = None  # No geometric sliding window in this strategy.
        if self.reps < 0 or min(self.max_epochs, self.block_maxiter, self.global_maxiter) < 0:
            raise ValueError("Circuit depth and iteration limits must be nonnegative")
        if not np.isfinite(energy_tol) or energy_tol <= 0:
            raise ValueError("energy_tol must be positive")
        if self.entanglement not in ("linear", "reverse_linear"):
            raise ValueError("Expected linear or reverse_linear")
        if any(g not in ("rx", "ry", "rz") for g in self.su2_gates):
            raise ValueError("Parameter-shift requires rx, ry, or rz rotations")
        if max_block_qubits is not None and max_block_qubits < 1:
            raise ValueError("max_block_qubits must be positive or None")
        self.max_block_qubits = max_block_qubits
        self.full_pauli = qiskit_hamiltonian.simplify(atol=0, rtol=0)
        if self.full_pauli.num_qubits != self.num_qubits:
            raise ValueError("Hamiltonian/register width mismatch")
        if np.any(np.abs(self.full_pauli.coeffs.imag) > 1e-12):
            raise ValueError("Hamiltonian must have real Pauli coefficients")
        # Derive both objectives from the same source, preventing mismatched Hs.
        self.hamiltonian = self._spin(self.full_pauli)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_qubits))
        for label, coeff in self.full_pauli.to_list():
            support = [q for q, p in enumerate(label[::-1]) if p != "I"]
            for a, b in combinations(support, 2):
                weight = self.graph.get_edge_data(a, b, {}).get("weight", 0.0)
                if abs(coeff):
                    self.graph.add_edge(a, b, weight=weight + float(abs(coeff)))
        self.blocks = self._partition() if blocks is None else [sorted(b) for b in blocks]
        flattened = [q for b in self.blocks for q in b]
        if any(not b for b in self.blocks) or sorted(flattened) != list(range(self.num_qubits)):
            raise ValueError("Blocks must cover each qubit exactly once")
        owner = {q: i for i, block in enumerate(self.blocks) for q in block}
        self.cross_edges = sorted((a, b) for a, b in self.graph.edges if owner[a] != owner[b])
        self.local_terms = [[] for _ in self.blocks]
        self.cross_terms = []
        for label, coeff in self.full_pauli.to_list():
            if coeff == 0:
                continue
            support = [q for q, p in enumerate(label[::-1]) if p != "I"]
            owners = {owner[q] for q in support}
            if len(owners) > 1:
                self.cross_terms.append((label, float(coeff.real)))
                continue
            index = next(iter(owners)) if owners else 0  # constant exactly once
            block = self.blocks[index]
            chars = [label[self.num_qubits - 1 - q] for q in block]
            chars += ["I"] * (max(2, len(block)) - len(block))
            self.local_terms[index].append(("".join(chars[::-1]), float(coeff.real)))
        self.local_models = []
        offset = 0
        for index, block in enumerate(self.blocks):
            width = max(2, len(block))
            terms = self.local_terms[index] or [("I" * width, 0.0)]
            pauli = SparsePauliOp.from_list(terms)
            count = len(block) * (self.reps + 1) * len(self.su2_gates)
            kernel = self._build_kernel([list(range(len(block)))], width)
            self.local_models.append(dict(kernel=kernel, hamiltonian=self._spin(pauli),
                pauli=pauli, qubits=block, width=width, count=count,
                constant=all(set(label) <= {"I"} for label, _ in terms),
                start=offset, stop=offset + count,
                parameter_indices=[(layer*self.num_qubits+q)*len(self.su2_gates)+gate
                    for layer in range(self.reps+1) for q in block
                    for gate in range(len(self.su2_gates))]))
            offset += count
        self.local_parameter_count = offset
        self.num_params = offset
        # Global parameter order is layer, physical qubit, rotation gate.
        # Partitions affect the initialization, never the global CX topology.
        self.kernel = self._build_kernel([list(range(self.num_qubits))], max(2, self.num_qubits))
        self.param_bounds = [(-np.pi, np.pi)] * self.num_params
        self.param_order = [(gate, q, layer, gate_index)
            for layer in range(self.reps + 1) for q in range(self.num_qubits)
            for gate_index, gate in enumerate(self.su2_gates)]
        self.qubit_to_params = {q: [] for q in range(self.num_qubits)}
        for index, (_, q, _, _) in enumerate(self.param_order):
            self.qubit_to_params[q].append(index)
        self.local_results = []
        self.global_results = []
        self.partition_summary = dict(
            blocks=self.blocks, local_circuit_qubits=[m["width"] for m in self.local_models],
            local_term_counts=[len(t) for t in self.local_terms],
            crossing_term_count=len(self.cross_terms),
            crossing_coefficient_l1=sum(abs(c) for _, c in self.cross_terms),
            cross_edges=[list(e) for e in self.cross_edges],
            local_parameter_count=offset, cross_parameter_count=0, global_parameter_count=self.num_params,
            max_block_qubits=max_block_qubits,
            independent_objectives_exactly_separable=not self.cross_terms,
            cross_gates="none", global_ansatz="nearest_neighbor_su2_v1",
            global_entanglement=self.entanglement, reps=self.reps,
            initialization="local best rotations mapped by qubit/layer/gate; not a product-state embedding",
        )

    @staticmethod
    def _spin(pauli):
        # Converter returns a scalar for constant-only objectives. observe needs
        # a SpinOperator even for zero/identity Hamiltonians.
        return qiskit_sparsepauliop_to_cudaq_spin(pauli) + 0.0 * cudaq.spin.i(0)

    def _get_block_param_indices(self, qubits):
        # Every rotation belongs to exactly one physical qubit.
        return sorted({i for q in qubits for i in self.qubit_to_params[q]})

    def _partition(self):
        communities = []
        for component in nx.connected_components(self.graph):
            graph = self.graph.subgraph(component)
            if graph.number_of_edges():
                communities.extend(nx.community.louvain_communities(graph, weight="weight", seed=self.seed))
            else:
                communities.extend({q} for q in component)
        pending, result = list(communities), []
        while pending:
            block = pending.pop()
            if self.max_block_qubits is None or len(block) <= self.max_block_qubits:
                result.append(sorted(block))
            else:
                # Bounded local workload; all cut terms are retained globally.
                left, right = nx.community.kernighan_lin_bisection(
                    self.graph.subgraph(sorted(block)), weight="weight", seed=self.seed)
                pending.extend((left, right))
        return sorted(result, key=lambda block: block[0])

    def _build_kernel(self, blocks, width):
        count = sum(len(b) for b in blocks) * (self.reps + 1) * len(self.su2_gates)
        kernel, *params = cudaq.make_kernel(*([float] * count))
        register = kernel.qalloc(width)
        index = 0
        for block in blocks:
            for layer in range(self.reps + 1):
                for q in block:
                    for gate in self.su2_gates:
                        getattr(kernel, gate)(params[index], register[q])
                        index += 1
                if layer < self.reps and len(block) > 1:
                    for a, b in pairs_for_pattern(len(block), layer, self.entanglement):
                        kernel.cx(register[block[a]], register[block[b]])
        return kernel

    def _minimize_model(self, params, model, phase, maxiter, qpu_id=0):
        params = np.asarray(params, dtype=float).copy()
        cache = {}
        best = [float("inf"), params.copy()]
        previous = [None]
        accepted_count = 0
        reason = "iteration_limit"

        def evaluate(x, kind="energy", index=None, shift=None):
            return self._evaluate_energy(x, qpu_id=qpu_id, kernel=model["kernel"],
                hamiltonian=model["hamiltonian"], evaluation_kind=kind,
                parameter_index=index, parameter_shift=shift)

        def cost(x):
            key = np.asarray(x).tobytes()
            if key in cache:
                return cache[key]
            energy = evaluate(x)
            cache[np.asarray(x).tobytes()] = energy
            if previous[0] is None:
                previous[0] = energy
            if energy < best[0]:
                best[:] = [energy, x.copy()]
            if phase == "global":
                self._remember_global(energy, x)
            return energy

        def gradient(x):
            grad = np.empty(len(x))
            for i in range(len(x)):
                plus, minus = x.copy(), x.copy()
                plus[i] += np.pi / 2
                minus[i] -= np.pi / 2
                grad[i] = (evaluate(plus, "gradient", i, np.pi / 2)
                           - evaluate(minus, "gradient", i, -np.pi / 2)) / 2
            return grad

        def callback(x):
            nonlocal accepted_count, reason
            key = np.asarray(x).tobytes()
            energy = cache[key] if key in cache else cost(x)
            improvement = previous[0] - energy
            previous[0] = energy
            accepted_count += 1
            stop = improvement < self.energy_tol
            self._record_iteration(energy, "early_stop" if stop else "accepted")
            cache.clear()
            cache[key] = energy
            if stop:
                reason = "energy_improvement_below_threshold"
                raise self.EarlyStop()

        started = time.perf_counter()
        with self._timed_optimizer(phase, model["qubits"], qpu_id):
            cost(params)
            if maxiter and not model.get("constant", False):
                try:
                    method = "L-BFGS-B" if self.use_gradient_optimizer else "COBYLA"
                    options = {"maxiter": maxiter}
                    if self.use_gradient_optimizer:
                        options.update(ftol=1e-12, gtol=1e-8)
                    else:
                        options.update(tol=1e-8)
                    result = minimize(cost, params, method=method,
                        jac=gradient if self.use_gradient_optimizer else None,
                        callback=callback, options=options)
                    reason = str(result.message)
                except self.EarlyStop:
                    pass
            elif model.get("constant", False):
                reason = "constant_local_objective"
        return dict(energy=float(best[0]), params=best[1], qubits=model["qubits"],
                    circuit_qubits=model["width"], accepted_iterations=accepted_count,
                    stop_reason=reason, duration_s=time.perf_counter() - started)

    def _remember_global(self, energy, params):
        self.energy_history.append(float(energy))
        self.iteration_results.append((float(energy), params.copy()))
        self.iteration_results.sort(key=lambda item: item[0])
        del self.iteration_results[6:]

    def _run_vqe(self, init_params=None):
        self.nfev = 0
        self.energy_history, self.iteration_results, self.epoch_summaries = [], [], []
        self.local_results, self.global_results = [], []
        rng = np.random.default_rng(self.seed)
        theta = rng.uniform(-np.pi, np.pi, self.num_params) if init_params is None else np.asarray(init_params, dtype=float).copy()
        if theta.shape != (self.num_params,) or not np.all(np.isfinite(theta)):
            raise ValueError(f"Expected {self.num_params} finite initial parameters")
        workers = min(self.num_parallel_workers, len(self.blocks))
        self.partition_summary["local_workers"] = workers
        print(f"Independent VQE: blocks={self.blocks}, crossing terms={len(self.cross_terms)}, workers={workers}")
        # Environment configuration is process-wide. Change only between phases,
        # after all workers have joined; use one safe rank bound for every local.
        configure_gpu_groups(self.num_gpus_detected, workers)
        local_width = max(m["width"] for m in self.local_models)
        self.partition_summary["local_mps_max_bond"] = configure_full_bond(
            local_width, stages=[dict(entanglement=self.entanglement, reps=self.reps)])
        results = [None] * len(self.blocks)

        def work(worker):
            for index in range(worker, len(self.blocks), workers):
                model = self.local_models[index]
                results[index] = self._minimize_model(theta[model["parameter_indices"]],
                    model, "local", self.block_maxiter, worker)

        try:
            # Compile serially; CUDA-Q's Python builder/JIT is not thread safe.
            for model in self.local_models:
                model["kernel"].compile()
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(work, range(workers)))
            else:
                work(0)
        finally:
            configure_gpu_groups(self.num_gpus_detected)
            self.partition_summary["global_mps_max_bond"] = configure_full_bond(
                max(2, self.num_qubits), stages=[dict(entanglement=self.entanglement, reps=self.reps)])
        self.local_results = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in result.items()} for result in results]
        for model, result in zip(self.local_models, results):
            theta[model["parameter_indices"]] = result["params"]
        self.warm_start_params = theta.copy()
        global_model = dict(kernel=self.kernel, hamiltonian=self.hamiltonian,
                            qubits=list(range(self.num_qubits)), width=max(2, self.num_qubits))
        with self._timed_optimizer("warm_start", global_model["qubits"], 0):
            energy = self._evaluate_energy(theta)
        self.warm_start_energy = energy
        self._remember_global(energy, theta)
        if self.global_refinement and self.max_epochs > 0:
            # All local solves have joined and their best parameters are assembled.
            # Invoke the global optimizer once; its callback supplies early stopping.
            self._timing_epoch = 0
            started, count, seconds = time.perf_counter(), self.nfev, self._evaluation_seconds
            before = energy
            result = self._minimize_model(theta, global_model, "global", self.global_maxiter)
            self.global_results.append({k: v for k, v in result.items() if k != "params"})
            # Keep the initial global state as a candidate under noisy optimization.
            energy, theta = self.iteration_results[0]
            self.epoch_summaries.append(dict(epoch=0, offset=None,
                energy_current=before, energy_refined=result["energy"], best_energy=energy,
                duration_s=time.perf_counter() - started,
                energy_evaluation_count=self.nfev - count,
                energy_evaluation_time_s=self._evaluation_seconds - seconds))
        best_energy, best_params = self.iteration_results[0]
        print(f"Warm-start full-H energy: {self.warm_start_energy:.6f}; best global energy: {best_energy:.6f}")
        return self.energy_history, best_params.copy(), self.kernel, list(self.iteration_results)
