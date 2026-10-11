"""Configuration for the local full-bond, finite-shot MPS backend."""

import os

TARGET = "tensornet-mps-mqpu-shots"


def configure_full_bond(num_qubits: int, stages=None) -> int:
    """Use exact circuit bounds without imposing a maximum register width.

    Starting from a product state, each CNOT crossing a cut multiplies its
    Schmidt rank by at most two. Nearest-neighbor linear circuits cross each
    cut once per repetition; alternating pairwise circuits cross it at most
    ceil(reps/2) times. Other patterns use the arbitrary-state bound, covering
    intermediate states produced by the backend's routing SWAPs too.

    The backend stores ranks as signed C ints. For bounds above that capacity,
    pass its largest representable rank instead of overflowing atoi or rejecting
    the register. Such circuits can run while their actual ranks fit the backend
    and available GPU memory; this is not unlimited physical capacity.
    """
    if num_qubits < 2:
        raise ValueError(f"{TARGET} requires at least two qubits")
    exponents = []
    for stage in stages or [{"entanglement": None}]:
        exponent = num_qubits // 2
        pattern = stage.get("entanglement")
        reps = stage.get("reps")
        if isinstance(pattern, str) and reps is not None:
            reps = int(reps)
            if reps < 0:
                raise ValueError("reps must be nonnegative")
            if pattern.lower() in ("linear", "reverse_linear"):
                exponent = min(exponent, reps)
            elif pattern.lower() == "pairwise":
                exponent = min(exponent, (reps + 1) // 2)
        exponents.append(exponent)
    exponent = max(exponents)
    bond = (1 << exponent) if exponent < 31 else 2**31 - 1
    os.environ["CUDAQ_MPS_MAX_BOND"] = str(bond)
    os.environ["CUDAQ_MPS_ABS_CUTOFF"] = "0"
    os.environ["CUDAQ_MPS_RELATIVE_CUTOFF"] = "0"
    return bond


def configure_gpu_groups(num_gpus: int, num_workers: int = 1) -> str:
    """Give concurrent blocks disjoint GPU groups for fixed-budget shots."""
    num_workers = max(1, min(num_workers, num_gpus))
    groups = [[primary] for primary in range(num_workers)]
    for gpu in range(num_workers, num_gpus):
        groups[(gpu - num_workers) % num_workers].append(gpu)
    specification = ";".join(
        ",".join(map(str, group)) for group in groups
    )
    os.environ["CUDAQ_MPS_HBATCH_GROUPS"] = specification
    return specification
