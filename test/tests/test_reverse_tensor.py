import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'backend/python_dependencies'))
from reverse_tensor import TensorCircuit


class TensorTests(unittest.TestCase):
    def circuit(self):
        # Non-neighbouring/reversed CX, complex constants, and a reused angle.
        return SimpleNamespace(width=4, operations=[
            ('h',(0,),None), ('x',(2,),None), ('s',(0,),None),
            ('ry',(1,),('parameter',0)), ('cx',(1,3),None),
            ('rx',(0,),('mul',('constant',-.7),('parameter',1))),
            ('cx',(3,0),None), ('sdg',(3,),None),
            ('rz',(2,),('add',('parameter',0),('constant',.4))),
            ('y',(1,),None), ('ry',(3,),('parameter',2))])

    def dense(self, x):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        q = QuantumCircuit(4)
        q.h(0); q.x(2); q.s(0); q.ry(x[0],1); q.cx(1,3)
        q.rx(-.7*x[1],0); q.cx(3,0); q.sdg(3)
        q.rz(x[0]+.4,2); q.y(1); q.ry(x[2],3)
        return Statevector(q)

    def test_nonlocal_complex_circuit_and_all_gradient_components(self):
        from qiskit.quantum_info import SparsePauliOp
        words = ['XXYZ', 'IYZX', 'ZIII', 'ZZZZ', 'IIII']
        coefficients = [.7, -.4, .2, -.9, 1.1]
        h = SparsePauliOp.from_list([(w[::-1], c) for w,c in zip(words,coefficients)])
        problem = TensorCircuit(self.circuit(), words, coefficients, device='cpu')
        x = np.array([.37,-.28,.53])
        def energy(v): return float(self.dense(v).expectation_value(h).real)
        actual, gradient = problem.value_gradient(x)
        self.assertAlmostEqual(actual, energy(x), places=12)
        for i in range(len(x)):
            a,b=x.copy(),x.copy(); a[i]-=1e-6; b[i]+=1e-6
            self.assertAlmostEqual(gradient[i],(energy(b)-energy(a))/2e-6,places=8)
        problem.chunk=1
        value, derivative=problem.value_gradient(x)
        self.assertAlmostEqual(value,actual,places=12)
        np.testing.assert_allclose(derivative,gradient,atol=1e-12)

    def test_reconstruction_chain_rule_through_moments_and_eigenvalue(self):
        import torch
        problem=TensorCircuit(self.circuit(),['ZIII','IIIZ','XXIX'],device='cpu')
        def objective(v):
            m=problem.moments(v)
            matrix=torch.stack((torch.stack((m[0]*m[1], .3*m[2])),
                                torch.stack((.3*m[2],m[1]+.7))))
            return torch.linalg.eigvalsh(matrix)[0]
        x=torch.tensor([.37,-.28,.53],dtype=torch.float64,requires_grad=True)
        gradient=torch.autograd.grad(objective(x),x)[0].numpy()
        for i in range(len(x)):
            a,b=x.detach().clone(),x.detach().clone();a[i]-=1e-6;b[i]+=1e-6
            self.assertAlmostEqual(gradient[i],float((objective(b)-objective(a))/2e-6),places=8)

    def test_idle_large_register_never_requires_dense_state(self):
        kernel=SimpleNamespace(width=97,operations=[('ry',(0,),('parameter',0)),('cx',(0,96),None)])
        problem=TensorCircuit(kernel,['Z'+'I'*96,'I'*96+'Z'],[.3,.7],device='cpu')
        energy,gradient=problem.value_gradient([.4])
        self.assertAlmostEqual(energy,np.cos(.4),places=12)
        self.assertAlmostEqual(gradient[0],-np.sin(.4),places=12)


if __name__ == '__main__': unittest.main()
