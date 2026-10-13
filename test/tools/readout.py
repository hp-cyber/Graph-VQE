"""Convert CUDA-Q measurement keys for the Qiskit-based protein decoder."""


def protein_probabilities(counts):
    # CUDA-Q puts q0 at the left; Qiskit labels put q0 at the right.
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Cannot decode an empty measurement histogram")
    return {bits[::-1]: count / total for bits, count in counts.items()}
