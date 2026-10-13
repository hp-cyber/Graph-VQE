"""GPU correctness/integration checks: ./run.sh validate_independent_vqe.py."""
import csv
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import cudaq
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector

import Main
from Qiskit_VQE import IndependentSubspaceVQE, CudaqStateCalculator
from Qiskit_VQE.mps_config import configure_full_bond


def solver_for(h, **kwargs):
    return IndependentSubspaceVQE(qiskit_hamiltonian=h, hamiltonian=None,
                                 num_qubits=h.num_qubits, **kwargs)


def reference_block(n, theta, reps, pattern):
    circuit = QuantumCircuit(n)
    index = 0
    for layer in range(reps + 1):
        for q in range(n):
            circuit.ry(float(theta[index]), q)
            circuit.rz(float(theta[index + 1]), q)
            index += 2
        if layer == reps or n == 1:
            continue
        pairs = [(i, i + 1) for i in range(n - 1)]
        if pattern == "reverse_linear":
            pairs.reverse()
        elif pattern == "circular":
            pairs.append((n - 1, 0))
        elif pattern == "sca":
            offset = layer % n
            if offset % 2 == 0:
                pairs = ([(i, i + 1) for i in range(offset - 1, -1, -1)]
                         + [(n - 1, 0)]
                         + [(i - 1, i) for i in range(n - 1, offset, -1)])
            else:
                pairs = ([(i + 1, i) for i in range(n - offset - 1, n - 1)]
                         + [(0, n - 1)]
                         + [(i + 1, i) for i in range(n - offset - 1)])
        for a, b in pairs:
            circuit.cx(a, b)
    return circuit


def reference_state(solver, theta):
    return Statevector.from_instruction(reference_block(solver.num_qubits,theta,solver.reps,solver.entanglement))


def reconstruct(solver):
    terms = list(solver.cross_terms)
    for block, local in zip(solver.blocks, solver.local_terms):
        for label, coeff in local:
            full = ["I"] * solver.num_qubits
            for local_q, global_q in enumerate(block):
                full[global_q] = label[-1 - local_q]
            terms.append(("".join(full[::-1]), coeff))
    return SparsePauliOp.from_list(terms)


def stopping_checks(solver):
    visited = []
    def minimize(fun, x0, callback, **kwargs):
        fun(x0)
        for position in (1.0, 1.05):
            accepted = x0.copy(); accepted[0] = position
            fun(accepted)
            rejected = x0.copy(); rejected[0] = 2
            fun(rejected)
            visited.append(position)
            callback(accepted)
        raise AssertionError("Did not stop immediately on improvement 0.5 < 1")

    def energy(x, **kwargs):
        return 99 if x[0] == 2 else 100 - 10 * x[0]

    for phase, model in (("local", solver.local_models[0]),
                         ("global", dict(kernel=solver.kernel, hamiltonian=solver.hamiltonian,
                                         width=solver.num_qubits, qubits=list(range(solver.num_qubits))))):
        visited.clear()
        count = model.get("count", solver.num_params)
        with patch("Qiskit_VQE.independent_vqe.minimize", minimize), \
             patch.object(solver, "_evaluate_energy", energy):
            result = solver._minimize_model(np.zeros(count), model, phase, 100)
        assert result["energy"] == 89.5 and visited == [1.0, 1.05]
        assert result["stop_reason"] == "energy_improvement_below_threshold"
    assert solver.energy_tol == 1 and solver.patience is None and solver.window is None
    print("PASS immediate stopping at local/global accepted iterates, ignoring rejected trials")


def edge_case_checks():
    h = SparsePauliOp.from_list([("III", 2), ("IIZ", -3), ("IZZ", -5), ("ZZI", -7)])
    solver = solver_for(h, blocks=[[0], [1, 2]], reps=0,
                        block_maxiter=0, max_epochs=0)
    theta = np.zeros(solver.num_params)
    for _ in range(2):
        trace, *_ = solver.run_vqe(theta)
        np.testing.assert_allclose(trace, [-13])  # Includes the crossing term.
        assert solver.nfev == len(solver.energy_evaluation_timings) == 3
        assert sum(r["energy_evaluation_count"] for r in solver.iteration_timings) == 3
    solver.max_epochs = 100
    calls = []

    def optimizer(x, model, phase, maxiter, qpu_id=0):
        if phase == "global":
            calls.append(phase)
            solver._remember_global(-23, x)
        return dict(energy=-23, params=x.copy(), qubits=model["qubits"],
                    circuit_qubits=model["width"], accepted_iterations=1,
                    stop_reason="energy_improvement_below_threshold", duration_s=0)

    with patch.object(solver, "_minimize_model", optimizer):
        solver.run_vqe(theta)
    assert calls == ["global"] and len(solver.epoch_summaries) == 1
    assert solver.global_results[0]["stop_reason"] == "energy_improvement_below_threshold"
    terms = []
    for q in range(96):
        label = ["I"] * 97
        label[q] = label[q + 1] = "Z"
        terms.append(("".join(label), -1))
    large = solver_for(SparsePauliOp.from_list(terms), reps=1)
    assert sorted(q for block in large.blocks for q in block) == list(range(97))
    assert configure_full_bond(97) == 2147483647
    print("PASS crossing energy, rerun resets, outer-loop stop and 97-qubit construction (not simulation)")


def main():
    rng = np.random.default_rng(41)
    # Noncontiguous blocks, singleton spectator, constant, X/Y/Z and crossing terms.
    h = SparsePauliOp.from_list([("IIIII", 2.3), ("IIZIZ", -3), ("IZIZI", -4),
        ("IIYIX", .7), ("XIIII", .2), ("IIIZZ", .8), ("IXXIY", .4)])
    report = []
    for pattern in ("linear", "reverse_linear"):
        solver = solver_for(h, blocks=[[0, 2], [1, 3], [4]], reps=3,
                            entanglement=pattern, max_epochs=1, block_maxiter=1, global_maxiter=1)
        assert len(solver.param_order) == solver.num_params
        assert solver._get_block_param_indices(range(h.num_qubits)) == list(range(solver.num_params))
        np.testing.assert_allclose(reconstruct(solver).to_matrix(), h.to_matrix(), atol=1e-12)
        configure_full_bond(h.num_qubits)
        theta = rng.uniform(-1, 1, solver.num_params)
        # Mapping local rotations initializes a different, full-register chain.
        assert solver.num_params == h.num_qubits*(solver.reps+1)*2
        assert solver.partition_summary['cross_parameter_count'] == 0
        mapped=sorted(i for model in solver.local_models for i in model['parameter_indices'])
        assert mapped==list(range(solver.num_params))
        solver.shots = -1
        expected = reference_state(solver, theta)
        exact_energy = solver._evaluate_energy(theta)
        np.testing.assert_allclose(exact_energy, expected.expectation_value(h).real, atol=1e-6)
        errors = []
        for index in (0, solver.num_params//2, solver.num_params - 1):
            plus, minus = theta.copy(), theta.copy()
            plus[index] += np.pi / 2; minus[index] -= np.pi / 2
            gradient = (solver._evaluate_energy(plus) - solver._evaluate_energy(minus)) / 2
            plus, minus = theta.copy(), theta.copy()
            plus[index] += 1e-5; minus[index] -= 1e-5
            difference = (reference_state(solver, plus).expectation_value(h).real
                          - reference_state(solver, minus).expectation_value(h).real) / 2e-5
            errors.append(abs(gradient - difference))
        assert max(errors) < 1e-6, errors
        for model in solver.local_models:
            local_theta = theta[model["parameter_indices"]]
            local_state = Statevector.from_instruction(reference_block(len(model["qubits"]),
                local_theta, solver.reps, pattern))
            if model["width"] > len(model["qubits"]):
                local_state = Statevector.from_label("0").tensor(local_state)
            actual_energy = solver._evaluate_energy(local_theta, kernel=model["kernel"], hamiltonian=model["hamiltonian"])
            np.testing.assert_allclose(actual_energy, local_state.expectation_value(model["pauli"]).real, atol=1e-6)
        if pattern == "linear":
            stopping_checks(solver)
        solver.shots = 1000
        with tempfile.TemporaryDirectory(prefix="independent-timing-") as directory:
            solver.timing_directory = directory
            trace, best, kernel, top = solver.run_vqe(theta)
            evaluations = list(csv.DictReader((Path(directory) / "energy_evaluation_timings.csv").open()))
            iterations = list(csv.DictReader((Path(directory) / "iteration_timings.csv").open()))
            assert len(evaluations) == solver.nfev
            assert all(r["status"] == "ok" and int(r["shots"]) == 1000 for r in evaluations)
            assert {r["phase"] for r in evaluations} == {"local", "warm_start", "global"}
            assert sum(int(r["energy_evaluation_count"]) for r in iterations) == solver.nfev
            np.testing.assert_allclose(sum(float(r["energy_evaluation_time_s"]) for r in iterations),
                                       solver._evaluation_seconds, atol=1e-9)
            assert all(float(r["duration_s"]) >= 0 for r in iterations + evaluations)
            if Main.num_gpus > 1:
                assert len({r["qpu_id"] for r in evaluations if r["phase"] == "local"}) > 1
            assert len(solver.local_results) == 3
            assert trace[0] == solver.warm_start_energy and top[0][0] <= solver.warm_start_energy
            np.testing.assert_array_equal(best, top[0][1])
            for model,result in zip(solver.local_models,solver.local_results):
                np.testing.assert_array_equal(solver.warm_start_params[model["parameter_indices"]],result["params"])
            probabilities = CudaqStateCalculator(kernel, shots=1000).get_probability_distribution(best)
            np.testing.assert_allclose(sum(probabilities.values()), 1, atol=1e-12)
            assert os.environ["CUDAQ_MPS_HBATCH_GROUPS"] == ",".join(map(str, range(Main.num_gpus)))
            report.append(dict(pattern=pattern, evaluations=solver.nfev,
                               maximum_gradient_error=max(errors), warm_start_energy=solver.warm_start_energy,
                               best_sampled_energy=top[0][0], partition=solver.partition_summary))
        print("PASS circuit, gradients, parallel locals, timing and readout:", pattern)

    # Real Hamiltonian graph partition, including splitting large communities.
    for limit in (1, 3, 12, None):
        solver = solver_for(h, max_block_qubits=limit)
        np.testing.assert_allclose(reconstruct(solver).to_matrix(), h.to_matrix(), atol=1e-12)
        assert limit is None or max(map(len, solver.blocks)) <= limit
    # Empty and identity-only local objectives still yield observable operators.
    for coefficient in (0, 2):
        constant = solver_for(SparsePauliOp.from_list([("III", coefficient)]),
                              reps=0, max_epochs=0, block_maxiter=1)
        trace, *_ = constant.run_vqe()
        np.testing.assert_allclose(trace, [coefficient], atol=1e-12)
        assert all(r["stop_reason"] == "constant_local_objective" for r in constant.local_results)
    print("PASS partition coverage, bounded locals, singleton and constant objectives")
    edge_case_checks()

    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="independent-protein-") as directory:
        try:
            os.chdir(directory)
            Main.predict_protein_structure("VKDRS", "independent_smoke", "cudaq", None, max_iter=1)
            summaries = list(Path("Result/process_data").glob("*/independent_smoke/stage_summary.json"))
            assert len(summaries) == 1
            for path in summaries:
                data = json.loads(path.read_text())
                assert data["strategy_type"] == "independent_hamiltonian_subspace"
                assert data["energy_tol"] == 1 and data["final_shots"] == 1000
                assert data["patience"] is None and data["smoothing_window"] is None
                assert max(data["partition"]["local_circuit_qubits"]) < 7
                assert any(p.stat().st_size for p in (path.parent / "Structures").glob("*.xyz"))
                report.append(dict(protein="VKDRS", strategy=data["strategy"],
                    evaluations=data["num_function_evaluations"], execution_time_s=data["execution_time_s"],
                    partition=data["partition"]))
        finally:
            os.chdir(original)
    Path("validation_independent_vqe.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS real-protein pipeline; report: validation_independent_vqe.json")


if __name__ == "__main__":
    main()
