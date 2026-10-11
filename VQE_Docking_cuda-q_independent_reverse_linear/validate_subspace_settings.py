"""Check immediate stopping and 1000-shot runs without docking outputs."""

import os
from unittest.mock import patch

import cudaq
import numpy as np

from Qiskit_VQE import SubspaceVQE
from Qiskit_VQE.mps_config import TARGET, configure_gpu_groups


def main():
    num_gpus = cudaq.num_available_gpus()
    os.environ["CUDAQ_MQPU_NGPUS"] = str(num_gpus)
    os.environ["CUDAQ_MPS_HBATCH_NGPUS"] = str(num_gpus)
    configure_gpu_groups(num_gpus)
    cudaq.set_target(TARGET)
    hamiltonian = cudaq.spin.z(0) + 0.5 * cudaq.spin.z(1) * cudaq.spin.z(3)

    def solver_for(pattern):
        return SubspaceVQE(
            hamiltonian, num_qubits=4, shots=1000, reps=1,
            entanglement=pattern, energy_tol=1, immediate_early_stopping=True,
            max_epochs=3, block_maxiter=3, global_maxiter=3,
            use_hamiltonian_blocking=True, use_gradient_optimizer=True,
            enable_parallel=False,
        )

    solver = solver_for("linear")
    assert solver.window is None and solver.patience is None
    assert solver.block_patience is solver.global_patience is solver.epoch_patience is None
    theta = np.zeros(solver.num_params)
    tested = []

    def controlled_minimize(fun, x0, *, callback, options, **kwargs):
        assert options["ftol"] < 1e-8  # Must not use the absolute threshold as relative ftol.
        fun(x0)
        for step in (1.0, 1.05):
            candidate = x0.copy()
            candidate[0] = step
            fun(candidate)
            tested.append(step)
            callback(candidate)
        raise AssertionError("A 0.5 improvement must stop immediately")

    # A 10-unit improvement must continue; the next 0.5 must stop. Check both
    # optimizers with a reproducible sequence independent of shot fluctuations.
    with patch("Qiskit_VQE.subspace_vqe.minimize", controlled_minimize), \
         patch.object(solver, "_evaluate_energy", side_effect=lambda p, **kw: 100 - 10 * p[0]):
        _, energy = solver._optimize_block(theta, [0], 100)
        assert energy == 89.5 and tested == [1.0, 1.05]
        tested.clear()
        _, energy = solver._global_refinement(theta)
        assert energy == 89.5 and tested == [1.0, 1.05]

    # Epoch improvement 0.5 must stop even though it establishes a new best.
    with patch.object(solver, "_evaluate_energy", side_effect=[100, 99.5]), \
         patch.object(solver, "_sequential_block_optimization", return_value=(theta, [99.5])):
        solver.global_refinement = False
        solver.run_vqe(theta)
        assert len(solver.epoch_summaries) == 1

    for pattern in ("linear", "reverse_linear", "circular", "sca"):
        solver = solver_for(pattern)
        solver.run_vqe()
        counts = dict(cudaq.sample(solver.kernel, *theta.tolist(), shots_count=1000).items())
        assert sum(counts.values()) == 1000
        assert solver.shots == 1000 and solver.window is None
        print(f"PASS: {pattern}, 1000 readout samples, immediate stopping")
    print("PASS: absolute threshold 1 at block/global/epoch levels without patience or smoothing")


if __name__ == "__main__":
    main()
