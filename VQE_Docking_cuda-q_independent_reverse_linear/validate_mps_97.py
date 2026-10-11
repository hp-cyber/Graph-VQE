"""Exercise 97-qubit linear circuits: ./run.sh validate_mps_97.py."""

import os

import cudaq
import numpy as np

from Qiskit_VQE.efficientsu2 import efficient_su2_cudaq
from Qiskit_VQE.mps_config import TARGET, configure_full_bond, configure_gpu_groups


def main():
    num_gpus = cudaq.num_available_gpus()
    os.environ["CUDAQ_MQPU_NGPUS"] = str(num_gpus)
    os.environ["CUDAQ_MPS_HBATCH_NGPUS"] = str(num_gpus)
    configure_gpu_groups(num_gpus)
    cudaq.set_target(TARGET)
    rng = np.random.default_rng(97)
    hamiltonian = (0.3 * cudaq.spin.z(0) - 0.7 * cudaq.spin.z(48)
                   + 0.2 * cudaq.spin.z(96)
                   + 0.1 * cudaq.spin.z(0) * cudaq.spin.z(96))
    for pattern in ("linear", "reverse_linear"):
        kernel, order, _ = efficient_su2_cudaq(97, reps=4, entanglement=pattern)
        params = rng.uniform(-1, 1, len(order)).tolist()
        assert configure_full_bond(97, [{"entanglement": pattern, "reps": 4}]) == 16
        exact = cudaq.observe(kernel, hamiltonian, *params).expectation()
        # Increasing the cap beyond the circuit's proven rank bound must not
        # change the ideal result; this also exercises larger numerical ranks.
        os.environ["CUDAQ_MPS_MAX_BOND"] = "32"
        reference = cudaq.observe(kernel, hamiltonian, *params).expectation()
        np.testing.assert_allclose(exact, reference, atol=1e-6, rtol=0)
        configure_full_bond(97, [{"entanglement": pattern, "reps": 4}])
        sampled = cudaq.observe(kernel, hamiltonian, *params, shots_count=8192).expectation()
        assert abs(sampled - exact) < 0.08
        counts = dict(cudaq.sample(kernel, *params, shots_count=8192).items())
        assert sum(counts.values()) == 8192
        assert all(len(bits) == 97 for bits in counts)
        print(f"97-qubit {pattern}: bond=16; difference from bond=32 "
              f"{abs(exact-reference):.3g}; shot error={abs(sampled-exact):.3g}")

        # Independent known-state check: zero rotations leave |0...0> intact.
        known = cudaq.observe(kernel, hamiltonian, *([0.0] * len(order))).expectation()
        np.testing.assert_allclose(known, -0.1, atol=1e-10, rtol=0)
    print("PASS: 97-qubit linear/reverse-linear circuits with four repetitions")


if __name__ == "__main__":
    main()
