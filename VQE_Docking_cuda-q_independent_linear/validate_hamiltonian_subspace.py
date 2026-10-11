"""Audit graph, gradient, update fusion, and a real protein on the GPU backend.

Run with ./run.sh validate_hamiltonian_subspace.py. Docking outputs are untouched.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cudaq
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
from Qiskit_VQE import SubspaceVQE, CudaqStateCalculator, qiskit_sparsepauliop_to_cudaq_spin
from Qiskit_VQE.efficientsu2 import pairs_for_pattern
from Qiskit_VQE.mps_config import TARGET, configure_full_bond, configure_gpu_groups


def reference_energy(params, hamiltonian, pattern, reps):
    n = hamiltonian.num_qubits
    circuit = QuantumCircuit(n)
    index = 0
    for layer in range(reps + 1):
        for qubit in range(n):
            circuit.ry(float(params[index]), qubit)
            circuit.rz(float(params[index + 1]), qubit)
            index += 2
        if layer < reps:
            for control, target in pairs_for_pattern(n, layer, pattern):
                circuit.cx(control, target)
    return float(Statevector.from_instruction(circuit).expectation_value(hamiltonian).real)


def regression_checks():
    solver = SubspaceVQE(cudaq.spin.z(0), num_qubits=2, reps=0,
                         enable_parallel=False, global_refinement=False, window_size=1,
                         max_epochs=2, immediate_early_stopping=True, energy_tol=1)
    theta = np.zeros(solver.num_params)
    # Each update separately lowers energy from 10 to 8; fused updates yield 11.
    # A stale last-block energy would falsely record the fused state as best.
    def energy(params, **kwargs):
        a, b = params[0], params[2]
        return 10 - 2 * a - 2 * b + 5 * a * b

    def optimize(params, qubits, maxiter, qpu_id=None):
        candidate = params.copy()
        candidate[2 * qubits[0]] = 1
        return candidate, energy(candidate)

    with patch.object(solver, "_evaluate_energy", energy), \
         patch.object(solver, "_optimize_block", optimize):
        _, best, _, top = solver.run_vqe(theta)
        assert solver.epoch_summaries[0]["energy_current"] == 11
        np.testing.assert_array_equal(best, theta)
        assert top[0][0] == 10  # Initial best must also appear in top results.
        assert len(solver.epoch_summaries) == 1

    # A line search probes x=2 (energy 99) but accepts x=1 (energy 90).
    # Stopping must compare accepted x=1, then x=1.05 (energy 89.5).
    accepted = []
    # Evaluate the final accepted point before delivering its callback.
    def optimizer(fun, x0, *, callback, **kwargs):
        fun(x0)
        first = x0.copy(); first[0] = 1
        fun(first)
        rejected = x0.copy(); rejected[0] = 2
        fun(rejected)
        accepted.append(1)
        callback(first)
        second = x0.copy(); second[0] = 1.05
        fun(second)
        accepted.append(1.05)
        callback(second)
        raise AssertionError("Missing immediate early stop")

    def objective(params, **kwargs):
        return 99 if params[0] == 2 else 100 - 10 * params[0]

    solver.use_gradient_optimizer = True
    with patch("Qiskit_VQE.subspace_vqe.minimize", optimizer), \
         patch.object(solver, "_evaluate_energy", objective):
        _, value = solver._optimize_block(theta, [0], 100)
        assert value == 89.5 and accepted == [1, 1.05]
        accepted.clear()
        _, value = solver._global_refinement(theta)
        assert value == 89.5 and accepted == [1, 1.05]
    print("PASS: fused energy, initial best, and accepted-iterate stopping regressions")


def main():
    num_gpus = cudaq.num_available_gpus()
    os.environ["CUDAQ_MQPU_NGPUS"] = str(num_gpus)
    os.environ["CUDAQ_MPS_HBATCH_NGPUS"] = str(num_gpus)
    configure_gpu_groups(num_gpus)
    cudaq.set_target(TARGET)
    regression_checks()

    problem = ProteinFoldingProblem(Peptide("VKDRS", [""] * 5),
                                    MiyazawaJerniganInteraction(), PenaltyParameters(10, 10, 10))
    hamiltonian = problem.qubit_op()
    spin = qiskit_sparsepauliop_to_cudaq_spin(hamiltonian)
    n = hamiltonian.num_qubits
    expected_edges = {}
    for word, coefficient in hamiltonian.to_list():
        active = [index for index, pauli in enumerate(reversed(word)) if pauli != "I"]
        for i, left in enumerate(active):
            for right in active[i + 1:]:
                edge = (left, right)
                expected_edges[edge] = expected_edges.get(edge, 0) + abs(coefficient)

    rng = np.random.default_rng(13)
    report = []
    for pattern in ("linear", "reverse_linear", "circular", "sca"):
        configure_gpu_groups(num_gpus)
        solver = SubspaceVQE(
            spin, num_qubits=n, shots=1000, reps=4, entanglement=pattern,
            use_hamiltonian_blocking=True, use_gradient_optimizer=True,
            immediate_early_stopping=True, energy_tol=1,
            max_epochs=2, block_maxiter=2, global_maxiter=2,
        )
        graph = solver._construct_interaction_graph()
        actual_edges = {tuple(sorted((u, v))): data["weight"] for u, v, data in graph.edges(data=True)}
        assert actual_edges.keys() == expected_edges.keys()
        for edge in expected_edges:
            np.testing.assert_allclose(actual_edges[edge], expected_edges[edge], atol=1e-12)
        blocks = solver._partition_hamiltonian_graph()
        assert sorted(q for block in blocks for q in block) == list(range(n))
        assert blocks == solver._partition_hamiltonian_graph()
        indices = [solver._get_block_param_indices(block) for block in blocks]
        assert sorted(p for block in indices for p in block) == list(range(solver.num_params))
        configure_full_bond(n, [{"entanglement": pattern, "reps": 4}])
        theta = rng.uniform(-1, 1, solver.num_params)
        solver.shots = -1
        energy = solver._evaluate_energy(theta)
        np.testing.assert_allclose(energy, reference_energy(theta, hamiltonian, pattern, 4), atol=1e-6)
        selected = [0, 1, solver.num_params // 2, solver.num_params - 1]
        gradient = solver._compute_block_gradient(theta, selected)
        finite_difference = []
        for index in selected:
            plus, minus = theta.copy(), theta.copy()
            plus[index] += 1e-5; minus[index] -= 1e-5
            finite_difference.append((reference_energy(plus, hamiltonian, pattern, 4)
                                      - reference_energy(minus, hamiltonian, pattern, 4)) / 2e-5)
        np.testing.assert_allclose(gradient, finite_difference, atol=1e-6, rtol=0)
        solver.shots = 1000
        trace, best, kernel, top = solver.run_vqe(theta)
        assert np.isfinite(trace).all() and np.isfinite(best).all()
        assert os.environ["CUDAQ_MPS_HBATCH_GROUPS"] == ",".join(map(str, range(num_gpus)))
        solver.shots = -1
        final_energy = solver._evaluate_energy(best)
        np.testing.assert_allclose(final_energy, reference_energy(best, hamiltonian, pattern, 4), atol=1e-6)
        probabilities = CudaqStateCalculator(kernel, shots=1000).get_probability_distribution(best)
        np.testing.assert_allclose(sum(probabilities.values()), 1, atol=1e-12)
        result = problem.interpret(probabilities)
        assert result is not None
        with tempfile.TemporaryDirectory(prefix="hamiltonian-subspace-") as directory:
            result.save_xyz_file(name="validation", path=directory)
            assert any(path.stat().st_size > 0 for path in Path(directory).glob("*.xyz"))
        report.append({"ansatz": pattern, "qubits": n, "blocks": blocks,
                       "epochs": len(solver.epoch_summaries), "evaluations": solver.nfev,
                       "initial_exact_energy": energy, "final_exact_energy": final_energy,
                       "gradient_max_error": float(np.max(np.abs(gradient - finite_difference)))})
        print("PASS protein strategy:", json.dumps(report[-1]))
    print("PASS: Hamiltonian-aware protein strategies on", num_gpus, "GPU(s)")
    Path("validation_hamiltonian_subspace.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
