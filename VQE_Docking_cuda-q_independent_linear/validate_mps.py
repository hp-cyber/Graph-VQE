"""GPU smoke check: ./run.sh validate_mps.py (no docking outputs written)."""

import os
from concurrent.futures import ThreadPoolExecutor

import cudaq
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from Qiskit_VQE import SubspaceVQE, CudaqStateCalculator
from Qiskit_VQE.cudaq import qiskit_sparsepauliop_to_cudaq_spin
from Qiskit_VQE.efficientsu2 import efficient_su2_cudaq, pairs_for_pattern
from Qiskit_VQE.mps_config import TARGET, configure_full_bond, configure_gpu_groups


def main():
    num_gpus = cudaq.num_available_gpus()
    os.environ["CUDAQ_MQPU_NGPUS"] = str(num_gpus)
    os.environ["CUDAQ_MPS_HBATCH_NGPUS"] = str(num_gpus)
    configure_gpu_groups(num_gpus)
    cudaq.set_target(TARGET)
    cudaq.set_random_seed(17)
    assert cudaq.get_target().num_qpus() == num_gpus
    assert configure_full_bond(6) == 8
    assert configure_full_bond(62) == 2**31 - 1
    assert configure_full_bond(97) == 2**31 - 1
    assert configure_full_bond(97, [{"entanglement": "linear", "reps": 4}]) == 16
    assert configure_full_bond(97, [{"entanglement": "reverse_linear", "reps": 4}]) == 16
    assert configure_full_bond(97, [{"entanglement": "pairwise", "reps": 4}]) == 4
    configure_full_bond(6)

    hamiltonian = SparsePauliOp.from_list([
        ("IIIIIZ", 0.3), ("IIIIZZ", -0.7), ("IIZIII", 0.2),
        ("IIIIIX", 0.15), ("IIIYII", -0.1), ("IIIIII", 0.4),
    ])
    spin = qiskit_sparsepauliop_to_cudaq_spin(hamiltonian)
    rng = np.random.default_rng(17)
    for pattern in ("linear", "reverse_linear", "circular", "full", "pairwise", "sca"):
        configure_full_bond(6, [{"entanglement": pattern, "reps": 2}])
        kernel, order, _ = efficient_su2_cudaq(6, reps=2, entanglement=pattern)
        params = rng.uniform(-1, 1, len(order)).tolist()
        circuit = QuantumCircuit(6)
        index = 0
        for layer in range(3):
            for qubit in range(6):
                circuit.ry(params[index], qubit)
                circuit.rz(params[index + 1], qubit)
                index += 2
            if layer < 2:
                for control, target in pairs_for_pattern(6, layer, pattern):
                    circuit.cx(control, target)
        expected = float(Statevector.from_instruction(circuit).expectation_value(hamiltonian).real)
        exact = cudaq.observe(kernel, spin, *params).expectation()
        np.testing.assert_allclose(exact, expected, atol=1e-7, rtol=0)
        measured = cudaq.observe(kernel, spin, *params, shots_count=8192).expectation()
        assert abs(measured - expected) < 0.08, (pattern, measured, expected)
        counts = dict(cudaq.sample(kernel, *params, shots_count=8192).items())
        assert sum(counts.values()) == 8192
        probabilities = CudaqStateCalculator(kernel, shots=8192).get_probability_distribution(params)
        np.testing.assert_allclose(sum(probabilities.values()), 1, atol=1e-12)
        print(f"{pattern}: exact error={abs(exact-expected):.3g}; shot error={abs(measured-expected):.3g}; 8192 samples")

    # Exercise the actual block optimizer's explicit QPU path concurrently.
    configure_gpu_groups(num_gpus, min(2, num_gpus))
    solver = SubspaceVQE(spin, num_qubits=6, reps=2, shots=8192,
                         max_epochs=1, block_maxiter=14, global_maxiter=1,
                         global_refinement=False)
    params = rng.uniform(-1, 1, solver.num_params)
    solver._evaluate_energy(params)  # Compile before concurrent requests.
    with ThreadPoolExecutor(max_workers=min(2, num_gpus)) as executor:
        results = list(executor.map(
            lambda qpu: solver._evaluate_energy(params, qpu_id=qpu),
            range(min(2, num_gpus)),
        ))
    assert all(np.isfinite(results))
    solver._parallel_block_optimization(params, [[0, 1], [2, 3], [4, 5]])
    assert os.environ["CUDAQ_MPS_HBATCH_GROUPS"] == ",".join(map(str, range(num_gpus)))
    print(f"PASS: full-bond energies, finite-shot sampling, and block scheduling on {num_gpus} GPU(s)")


if __name__ == "__main__":
    main()
