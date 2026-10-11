"""Exact reverse-mode tensor contractions for arbitrary recorded circuit topology.

No dense full-register state or finite differences are used. Term chunks bound
the reverse tape; contraction plans are reused. Large exact contractions can
still exceed GPU memory, in which case the job reports the resource limitation.
"""
import os
import numpy as np


def angle_value(expr, x):
    kind, *args = expr
    if kind == 'parameter': return x[args[0]]
    if kind == 'constant': return x.new_tensor(args[0])
    a, b = (angle_value(arg, x) for arg in args)
    return {'add': lambda: a+b, 'sub': lambda: a-b,
            'mul': lambda: a*b, 'div': lambda: a/b}[kind]()


class TensorCircuit:
    engine = 'reverse_tensor_network'

    def __init__(self, kernel, words, coefficients=None, device=None):
        import torch
        import opt_einsum as oe
        self.torch, self.oe = torch, oe
        self.kernel, self.words = kernel, list(words)
        self.coefficients = np.ones(len(words)) if coefficients is None else np.asarray(coefficients, dtype=float)
        self.device = device or os.environ.get('VQE_REVERSE_DEVICE', 'cuda')
        self.chunk = int(os.environ.get('VQE_REVERSE_TERM_BATCH', '32'))
        if self.chunk < 1: raise ValueError('VQE_REVERSE_TERM_BATCH must be positive')
        self.plans = {}
        t = torch
        self.constants = {name: t.tensor(matrix, dtype=t.complex128, device=self.device) for name, matrix in {
            'i': [[1,0],[0,1]], 'x': [[0,1],[1,0]], 'y': [[0,-1j],[1j,0]],
            'z': [[1,0],[0,-1]], 'h': np.array([[1,1],[1,-1]])/np.sqrt(2),
            's': [[1,0],[0,1j]], 'sdg': [[1,0],[0,-1j]],
            'cx': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).reshape(2,2,2,2)
        }.items()}
        self.paulis = t.stack([self.constants[p] for p in ('i','x','y','z')])
        self.labels = [[q] for q in range(kernel.width)]
        self.ends = list(range(kernel.width))
        cursor = kernel.width
        for name, qubits, expr in kernel.operations:
            inputs = [self.ends[q] for q in qubits]
            outputs = list(range(cursor, cursor+len(qubits)))
            cursor += len(qubits)
            self.labels.append(outputs+inputs)
            for q, wire in zip(qubits, outputs): self.ends[q] = wire
        self.offset = cursor

    def _operands(self, x):
        t = self.torch
        zero = t.tensor([1.,0.], dtype=t.complex128, device=self.device)
        operands = [zero]*self.kernel.width
        for name, qubits, expr in self.kernel.operations:
            if expr is None:
                operands.append(self.constants[name])
            else:
                angle = angle_value(expr, x)
                # exp(-i theta P / 2), valid for RX/RY/RZ and shared expressions.
                operands.append(t.cos(angle/2)*self.constants['i'] -
                                1j*t.sin(angle/2)*self.constants[name[1]])
        return operands

    def _plan(self, count):
        if count not in self.plans:
            batch = 2*self.offset
            labels = self.labels + [[i+self.offset for i in label] for label in self.labels]
            labels += [[batch, q+self.offset, q] for q in self.ends]
            shapes = [(2,)*len(label) for label in self.labels]*2
            shapes += [(count,2,2)]*self.kernel.width
            equation = ','.join(''.join(self.oe.get_symbol(i) for i in label) for label in labels)
            equation += '->'+self.oe.get_symbol(batch)
            path, info = self.oe.contract_path(equation, *shapes, shapes=True, optimize='greedy')
            if str(self.device).startswith('cuda'):
                free, _ = self.torch.cuda.mem_get_info(self.device)
                if int(info.largest_intermediate)*16*4 > free:
                    raise MemoryError('Exact reverse contraction exceeds available GPU workspace; no truncation or finite-difference fallback was applied')
            self.plans[count] = self.oe.contract_expression(equation, *shapes, optimize=path)
        return self.plans[count]

    def _values(self, x, words):
        t = self.torch
        operands = self._operands(x)
        operands += [a.conj() for a in operands]
        for q in range(self.kernel.width):
            indices = t.tensor(['IXYZ'.index(w[q]) for w in words], device=self.device)
            operands.append(self.paulis[indices])
        return self._plan(len(words))(*operands, backend='torch').real

    def values(self, parameters):
        t = self.torch
        x = t.as_tensor(np.asarray(parameters), dtype=t.float64, device=self.device)
        with t.no_grad():
            values = [self._values(x, self.words[i:i+self.chunk]).cpu().numpy()
                      for i in range(0, len(self.words), self.chunk)]
        return np.concatenate(values) if values else np.zeros(0)

    def value_gradient(self, parameters, coefficients=None):
        t = self.torch
        coefficients = self.coefficients if coefficients is None else coefficients
        x = t.tensor(np.asarray(parameters), dtype=t.float64, device=self.device, requires_grad=True)
        total, gradient = 0., t.zeros_like(x)
        for i in range(0, len(self.words), self.chunk):
            weights = t.as_tensor(coefficients[i:i+self.chunk], dtype=t.float64, device=self.device)
            value = (self._values(x, self.words[i:i+self.chunk])*weights).sum()
            total += float(value.detach().cpu())
            if value.requires_grad:
                derivative = t.autograd.grad(value, x, allow_unused=True)[0]
                if derivative is not None: gradient += derivative
        return total, gradient.detach().cpu().numpy()

    def moments(self, x):
        """Vector moments with a reverse VJP, without retaining all term tapes."""
        t, circuit = self.torch, self
        class Moments(t.autograd.Function):
            @staticmethod
            def forward(ctx, parameters):
                ctx.save_for_backward(parameters)
                values = circuit.values(parameters.detach().cpu().numpy())
                return parameters.new_tensor(values)
            @staticmethod
            def backward(ctx, weights):
                parameters, = ctx.saved_tensors
                with t.enable_grad():
                    _, derivative = circuit.value_gradient(parameters.detach().cpu().numpy(),
                                                           weights.detach().cpu().numpy())
                return parameters.new_tensor(derivative)
        return Moments.apply(x)
