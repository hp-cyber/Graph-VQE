"""Reverse differentiation of the original partition/forging scalar objectives."""
import os
import numpy as np
from reverse_mps import make_problem
from reverse_mps import recorded_problem as TensorCircuit


def circuit_problem(kernel, hamiltonian, base=None, indices=None):
    problem = make_problem(kernel, hamiltonian)
    if problem is None: raise ValueError('Circuit has no reverse gradient representation')
    if indices is None: return problem
    fixed, selected = np.asarray(base, dtype=float).copy(), np.asarray(indices, dtype=int).copy()
    class Projection:
        engine = problem.engine
        def value_gradient(self, parameters):
            full = fixed.copy()
            full[selected] = parameters
            energy, gradient = problem.value_gradient(full)
            return energy, gradient[selected]
    return Projection()


class ReconstructedProblem:
    engine = 'reverse_reconstruction'

    def __init__(self, solver):
        import torch
        self.torch, self.solver = torch, solver
        self.device = os.environ.get('VQE_REVERSE_DEVICE', 'cuda')
        self.coefficients = torch.tensor([t.coefficient for t in solver.terms],
                                         dtype=torch.float64, device=self.device)
        self.cache = {}

    def local_moments(self, kernel, block, parameters):
        key = (id(kernel), tuple(block))
        if key not in self.cache:
            labels = [''.join('Z' if q in term.qubits else 'I' for q in block).ljust(kernel.width, 'I')
                      for term in self.solver.terms]
            words = list(dict.fromkeys(labels))
            lookup = {word: i for i, word in enumerate(words)}
            indices = self.torch.tensor([lookup[w] for w in labels], device=self.device)
            self.cache[key] = TensorCircuit(kernel, words, device=self.device), indices
        circuit, indices = self.cache[key]
        return circuit.moments(parameters)[indices]

    def partitioned_energy(self, parameters):
        s = self.solver
        factors = [self.local_moments(kernel, block, parameters[s.offsets[i]:s.offsets[i+1]])
                   for i, (kernel, block) in enumerate(zip(s.kernels, s.blocks))]
        return (self.torch.stack(factors).prod(dim=0)*self.coefficients).sum()

    def forged_energy(self, parameters):
        s, t = self.solver, self.torch
        count_a, _ = s._side_parameter_counts()
        moments = {}
        def element(side, left, right):
            block = s.partition_a if side == 'a' else s.partition_b
            bases = s.basis_a if side == 'a' else s.basis_b
            theta = parameters[:count_a] if side == 'a' else parameters[count_a:]
            def values(descriptor):
                key = (side, descriptor)
                if key not in moments:
                    moments[key] = self.local_moments(s._kernel(side, descriptor), block, theta)
                return moments[key]
            if left == right: return values(('basis', bases[left])).to(t.complex128)
            v = [values(('pair', bases[left], bases[right], phase)) for phase in range(4)]
            return ((v[0]-v[2]) + 1j*(v[3]-v[1]))/2
        entries = {}
        for left in range(s.schmidt_rank):
            for right in range(left, s.schmidt_rank):
                entries[left, right] = (self.coefficients*(element('a', left, right)*
                                                          element('b', left, right)).real).sum()
        effective = t.stack([t.stack([entries[min(i,j), max(i,j)] for j in range(s.schmidt_rank)])
                             for i in range(s.schmidt_rank)])
        # Eigenvalue-only differentiation gives the Hellmann-Feynman derivative;
        # at degeneracy Torch selects a ground-space subgradient.
        return t.linalg.eigvalsh(effective)[0]

    def value_gradient(self, parameters):
        t = self.torch
        x = t.tensor(np.asarray(parameters), dtype=t.float64, device=self.device, requires_grad=True)
        energy = self.partitioned_energy(x) if type(self.solver).__name__ == 'PartitionedVQE' else self.forged_energy(x)
        gradient = t.autograd.grad(energy, x, allow_unused=True)[0] if energy.requires_grad else None
        if gradient is None: gradient = t.zeros_like(x)
        return float(energy.detach().cpu()), gradient.detach().cpu().numpy()


def published_problem(solver):
    name = type(solver).__name__
    if name == 'ExtendedClassicallySplitVQE':
        return circuit_problem(solver.kernel, solver.hamiltonian)
    if name in ('PartitionedVQE', 'EntanglementForgingVQE', 'EDEFAdaptVQE'):
        return ReconstructedProblem(solver)
    raise ValueError(f'Unsupported published reverse objective: {name}')
