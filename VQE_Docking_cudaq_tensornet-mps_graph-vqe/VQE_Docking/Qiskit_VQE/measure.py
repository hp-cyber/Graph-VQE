# --*-- conding:utf-8 --*--
# @Time : 11/13/24 12:08 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : measure.py
# --*-- coding:utf-8 --*--
# @File : measure_cudaq.py

from __future__ import annotations
import cudaq
from typing import Dict, Iterable, List
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from typing import Dict
import cudaq

class StateCalculator:
    def __init__(self,service ,min_qubit_num, ansatz: QuantumCircuit):
        """
        Initialize the ProbabilityDistributionCalculator with a predefined ansatz circuit.
        """
        self.service = service
        self.ansatz = ansatz
        # self.backend = self._select_backend(min_quits=min_qubit_num)
        self.min_qubits = min_qubit_num

    # def _select_backend(self, min_quits):
    #     """
    #             Selects the least busy IBM Quantum backend with enough qubits and returns it.
    #             This ensures that the quantum job is processed faster by using a backend with fewer queued jobs.
    #
    #             Returns:
    #             - backend: The IBM Quantum backend with the least busy queue.
    #     """
    #     backend = self.service.least_busy(simulator=False, operational=True, min_num_qubits=min_quits)
    #     return backend

    def get_probability_distribution(self, optimized_params) -> Dict:
        """
        Calculate the probability distribution by assigning optimized parameters to the ansatz,
        executing the circuit, and returning the measurement results.

        Parameters:
        optimized_params: The parameters to assign to the ansatz circuit.

        Returns:
        Dict: A dictionary with binary strings as keys and their probabilities as values.
        """
        circuit = self.ansatz.assign_parameters(optimized_params)
        circuit.measure_all()

        backend = self.service.least_busy(simulator=False, operational=True, min_num_qubits=self.min_qubits)

        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_circuit = pm.run(circuit)

        sampler = Sampler(backend,options={"default_shots": 100000})
        job = sampler.run([isa_circuit])
        result = job.result()

        threshold = 0 #

        pub_result = result[0]

        counts = pub_result.data.meas.get_counts()

        # total_shots = sum(counts.values())
        measure_result = {key: value for key, value in counts.items() if value > threshold}

        return measure_result


class CudaqStateCalculator:
    """
    Compute measurement probabilities for a CUDA-Q ansatz kernel.
    The kernel should be parameterized via `cudaq.make_kernel(list)` or accept a plain list of params.
    """

    def __init__(self, kernel, shots: int = 100000):
        """
        Parameters
        ----------
        kernel : CUDA-Q kernel
            A CUDA-Q kernel (possibly parameterized) that prepares the state to be measured.
            Example: built with `cudaq.make_kernel(list)` inside your EfficientSU2 builder.
        shots : int
            Number of shots to sample. Use -1 for noiseless statevector simulation.
        """
        self.kernel = kernel
        self.shots = shots

    def get_probability_distribution(self, optimized_params: Iterable[float], threshold: float = 0) -> Dict[str, float]:
        """
        Get probability distribution from the ansatz with optimized parameters.

        For shots=-1: Uses statevector to get exact probabilities (noiseless)
        For shots>0: Uses sampling to get empirical probabilities (with shot noise)

        Parameters
        ----------
        optimized_params : array-like
            Optimized circuit parameters
        threshold : float
            Minimum probability/count to include in output (default: 0)

        Returns
        -------
        Dict[str, float]
            Dictionary mapping bitstrings to probabilities
        """
        # Convert parameters to flat list to avoid shape issues with numpy arrays
        import numpy as np
        if isinstance(optimized_params, np.ndarray):
            params_list = optimized_params.flatten().tolist()
        else:
            params_list = list(optimized_params)

        if self.shots == -1:
            # Noiseless mode: get statevector and compute exact probabilities
            try:
                state = cudaq.get_state(self.kernel, *params_list)
            except TypeError:
                state = cudaq.get_state(self.kernel, params_list)

            # Extract amplitudes and compute probabilities
            # state is a cudaq.State object with amplitude information
            probs = {}

            # Try multiple methods to extract state vector data
            amplitudes = None
            n_qubits = None

            # Method 1: Try direct array access (newer CUDA-Q versions)
            if hasattr(state, '__len__') and hasattr(state, '__getitem__'):
                try:
                    n_states = len(state)
                    n_qubits = n_states.bit_length() - 1
                    amplitudes = [state[i] for i in range(n_states)]
                except Exception:
                    pass

            # Method 2: Try iteration (some CUDA-Q versions)
            if amplitudes is None and hasattr(state, '__iter__'):
                try:
                    amplitudes = list(state)
                    n_qubits = len(bin(len(amplitudes)-1)[2:])
                except Exception:
                    pass

            # Method 3: Try to_numpy if available
            if amplitudes is None and hasattr(state, 'to_numpy'):
                try:
                    amplitudes = state.to_numpy()
                    n_qubits = len(bin(len(amplitudes)-1)[2:])
                except Exception:
                    pass

            # Method 4: Try get_data() method (some versions)
            if amplitudes is None and hasattr(state, 'get_data'):
                try:
                    amplitudes = state.get_data()
                    n_qubits = len(bin(len(amplitudes)-1)[2:])
                except Exception:
                    pass

            if amplitudes is None:
                raise TypeError(
                    f"Cannot extract amplitudes from CUDA-Q state object. "
                    f"Available methods: {[m for m in dir(state) if not m.startswith('_')]}"
                )

            # Compute probabilities from amplitudes.
            # CUDA-Q uses big-endian bitstrings (qubit 0 = leftmost/MSB), but
            # the Qiskit-based interpret() expects little-endian (qubit 0 =
            # rightmost/LSB).  Reverse each bitstring so downstream protein
            # structure decoding sees the correct qubit ordering.
            for i, amplitude in enumerate(amplitudes):
                prob = abs(amplitude) ** 2
                if prob > threshold:
                    bitstring = format(i, f'0{n_qubits}b')[::-1]
                    probs[bitstring] = prob

            return probs
        else:
            # Sampling mode: use cudaq.sample for shot-based measurement
            try:
                result = cudaq.sample(self.kernel, *params_list, shots_count=self.shots)
            except TypeError:
                result = cudaq.sample(self.kernel, params_list, shots_count=self.shots)

            counts = None

            # case A: result.counts is a dict
            if hasattr(result, "counts") and not callable(getattr(result, "counts")):
                counts = dict(result.counts)

            # case B: result.counts() returns a dict
            elif hasattr(result, "counts") and callable(getattr(result, "counts")):
                counts = dict(result.counts())

            # case C: result.to_dict() exists
            elif hasattr(result, "to_dict") and callable(getattr(result, "to_dict")):
                counts = dict(result.to_dict())

            # case D: last resort – build from bitstrings using result.count(bstr) if available
            elif hasattr(result, "count") and callable(getattr(result, "count")):
                # Try to infer bit-width from the kernel argument (best effort: probe a few strings)
                # We'll gather all bitstrings that have nonzero counts up to a reasonable width guess.
                # If your kernel has known qubit count, you can pass it in instead.
                # Here we probe by asking result.count for observed keys via most_probable if present.
                bitstrings = set()
                if hasattr(result, "most_probable") and callable(getattr(result, "most_probable")):
                    mp = result.most_probable()
                    if isinstance(mp, str):
                        n = len(mp)
                        # check all bitstrings with nonzero count at Hamming distance 0/1/2 from mp (cheap)
                        candidates = [mp]
                        for i in range(n):
                            b = list(mp)
                            b[i] = "1" if b[i] == "0" else "0"
                            candidates.append("".join(b))
                        for c in candidates:
                            if result.count(c) > 0:
                                bitstrings.add(c)
                # If we didn't find any, just keep mp if it exists
                if not bitstrings and "mp" in locals() and isinstance(mp, str) and result.count(mp) > 0:
                    bitstrings.add(mp)
                counts = {b: result.count(b) for b in bitstrings}

            else:
                raise TypeError("Unsupported CUDA-Q sampling result type; cannot extract counts.")

            if threshold > 0:
                counts = {b: c for b, c in counts.items() if c > threshold}

            total = sum(counts.values()) if counts else 1
            # Reverse bitstrings: CUDA-Q returns big-endian (qubit 0 = leftmost)
            # but Qiskit's interpret() expects little-endian (qubit 0 = rightmost).
            probs = {b[::-1]: c / total for b, c in counts.items()}
            return probs

