"""Record the actual scalar-parameter CUDA-Q builders for reverse differentiation.

Every gate is still emitted into the native kernel. The tape contains parameter
expressions, including constants and reused parameters, rather than sampled angles.
"""
from dataclasses import dataclass


def native(value):
    return value.native if isinstance(value, (Parameter, Qubit, Kernel)) else value


def expression(value):
    return value.expression if isinstance(value, Parameter) else ('constant', float(value))


class Parameter:
    def __init__(self, value, expr):
        self.native, self.expression = value, expr

    def _binary(self, other, name, operation):
        return Parameter(operation(self.native, native(other)), (name, self.expression, expression(other)))

    def __add__(self, other): return self._binary(other, 'add', lambda a, b: a+b)
    def __sub__(self, other): return self._binary(other, 'sub', lambda a, b: a-b)
    def __mul__(self, other): return self._binary(other, 'mul', lambda a, b: a*b)
    def __truediv__(self, other): return self._binary(other, 'div', lambda a, b: a/b)
    def __neg__(self): return self * -1.
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return -self + other


@dataclass
class Qubit:
    native: object
    index: int


class Kernel:
    def __init__(self, kernel, parameters):
        self.native = kernel
        self.parameters = [Parameter(p, ('parameter', i)) for i, p in enumerate(parameters)]
        self.width, self.operations = 0, []

    def qalloc(self, count):
        if self.width:
            raise ValueError('Reverse circuit tape expects one register allocation')
        self.width = int(count)
        qubits = self.native.qalloc(count)
        return [Qubit(qubits[i], i) for i in range(count)]

    def argument(self, index): return self.parameters[index]
    def __str__(self): return str(self.native)

    def __getattr__(self, name):
        if name in ('rx', 'ry', 'rz'):
            def rotation(angle, qubit):
                self.operations.append((name, (qubit.index,), expression(angle)))
                return getattr(self.native, name)(native(angle), native(qubit))
            return rotation
        if name in ('h', 'x', 'y', 'z', 's', 'sdg', 'cx'):
            def gate(*qubits):
                self.operations.append((name, tuple(q.index for q in qubits), None))
                return getattr(self.native, name)(*(native(q) for q in qubits))
            return gate
        # Do not silently omit an unsupported quantum gate from the tape.
        if name in ('name', 'arguments', 'argument_count', 'compile'):
            return getattr(self.native, name)
        raise AttributeError(f'Unsupported reverse-tape kernel operation: {name}')


def install(cudaq):
    original = cudaq.make_kernel
    def make_kernel(*types):
        if any(t is not float for t in types):
            raise TypeError('Reverse circuit tape requires scalar float parameters')
        made = original(*types)
        inner, parameters = (made[0], list(made[1:])) if isinstance(made, tuple) else (made, [])
        kernel = Kernel(inner, parameters)
        return (kernel, *kernel.parameters) if types else kernel
    cudaq.make_kernel = make_kernel
