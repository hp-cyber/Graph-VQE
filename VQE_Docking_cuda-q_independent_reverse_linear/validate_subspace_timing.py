"""Validate timing records: ./run.sh validate_subspace_timing.py."""

import csv
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import cudaq
import numpy as np

from Qiskit_VQE import SubspaceVQE
from Qiskit_VQE.mps_config import TARGET, configure_gpu_groups


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    num_gpus = cudaq.num_available_gpus()
    os.environ["CUDAQ_MQPU_NGPUS"] = str(num_gpus)
    os.environ["CUDAQ_MPS_HBATCH_NGPUS"] = str(num_gpus)
    configure_gpu_groups(num_gpus)
    cudaq.set_target(TARGET)
    hamiltonian = cudaq.spin.z(0) * cudaq.spin.z(1) + 0.5 * cudaq.spin.z(2) * cudaq.spin.z(3)
    with tempfile.TemporaryDirectory(prefix="subspace-timing-") as directory:
        solver = SubspaceVQE(
            hamiltonian, num_qubits=4, shots=1000, reps=1, entanglement="linear",
            max_epochs=1, block_maxiter=1, global_maxiter=1,
            use_hamiltonian_blocking=True, use_gradient_optimizer=True,
            immediate_early_stopping=True, energy_tol=1, timing_directory=directory,
        )
        solver.run_vqe()
        evaluations = read_rows(Path(directory) / "energy_evaluation_timings.csv")
        iterations = read_rows(Path(directory) / "iteration_timings.csv")
        assert len(evaluations) == solver.nfev
        assert sorted(int(row["evaluation_id"]) for row in evaluations) == list(range(1, solver.nfev + 1))
        assert all(row["status"] == "ok" and int(row["shots"]) == 1000 for row in evaluations)
        for row in evaluations:
            assert float(row["duration_s"]) >= 0
            assert float(row["total_s"]) >= float(row["duration_s"])
            assert float(row["lock_wait_s"]) >= 0
            assert float(row["start_offset_s"]) >= 0
        assert any(row["kind"] == "gradient" and row["parameter_index"] for row in evaluations)
        assert any(row["phase"] == "global" for row in iterations)
        assert any(row["phase"] == "block" for row in iterations)
        assert any(row["status"] == "accepted" for row in iterations)
        if num_gpus > 1:
            assert {int(row["qpu_id"]) for row in evaluations if row["phase"] == "block"} == {0, 1}
        optimizer_evaluations = [row for row in evaluations if row["phase"] in ("block", "global")]
        assert sum(int(row["energy_evaluation_count"]) for row in iterations) == len(optimizer_evaluations)
        np.testing.assert_allclose(
            sum(float(row["energy_evaluation_time_s"]) for row in iterations),
            sum(float(row["duration_s"]) for row in optimizer_evaluations), atol=1e-9,
        )
        summary = solver.epoch_summaries[0]
        assert summary["duration_s"] > 0
        assert summary["energy_evaluation_count"] == len([row for row in evaluations if row["epoch"] == "0"])
        np.testing.assert_allclose(solver._evaluation_seconds,
                                   sum(float(row["duration_s"]) for row in evaluations), atol=1e-9)
        print(f"PASS: {len(evaluations)} evaluations, {len(iterations)} optimizer intervals on {num_gpus} GPU(s)")

        # Repeated runs must reset IDs and replace previous CSV data.
        solver.max_epochs = 0
        solver.run_vqe()
        assert len(read_rows(Path(directory) / "energy_evaluation_timings.csv")) == 1
        assert read_rows(Path(directory) / "iteration_timings.csv") == []

    # Backend failures must be recorded and release the sequential GPU lock.
    solver = SubspaceVQE(hamiltonian, num_qubits=4, enable_parallel=False)
    with patch.object(cudaq, "observe_async", side_effect=RuntimeError("timing failure test")):
        try:
            solver._evaluate_energy(np.zeros(solver.num_params))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Missing backend error")
    assert solver.energy_evaluation_timings[-1]["status"] == "error"
    assert solver.nfev == 0
    assert solver._gpu_lock.acquire(blocking=False)
    solver._gpu_lock.release()
    print("PASS: repeat-run reset and failed evaluation timing")


if __name__ == "__main__":
    main()
