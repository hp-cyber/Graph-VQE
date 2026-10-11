import sys,os,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from reverse_mps import ReverseMPS,register,_REGISTRY,for_objective

class ReverseTests(unittest.TestCase):
    def test_both_cx_orders_registered_and_gradients_match_dense(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, SparsePauliOp
        n, reps = 3, 2
        order = [(f't{layer}_{q}_{g}', q, layer, g)
                 for layer in range(reps+1) for q in range(n) for g in range(2)]
        words, coeffs = ['XYZ', 'ZZI', 'IYX', 'III'], [.3, -.7, .2, .4]
        h = SparsePauliOp.from_list([(w[::-1], c) for w, c in zip(words, coeffs)])
        x = np.random.default_rng(52).normal(size=len(order))
        energies = []
        for pattern in ('linear', 'reverse_linear'):
            k = object()
            edges = [(0, 1), (1, 2)]
            if pattern == 'reverse_linear': edges.reverse()
            register(k, n, reps, ('ry', 'rz'), pattern, None, order, [edges]*reps, True)
            self.assertIn(id(k), _REGISTRY)
            spec = _REGISTRY[id(k)][1]
            p = ReverseMPS(spec, words, coeffs, device='cpu')
            def dense(v):
                circuit = QuantumCircuit(n)
                for layer in range(reps+1):
                    for q in range(n):
                        i = (layer*n+q)*2
                        circuit.ry(v[i], q); circuit.rz(v[i+1], q)
                    if layer < reps:
                        for c, t in edges: circuit.cx(c, t)
                return float(Statevector(circuit).expectation_value(h).real)
            energy, gradient = p.value_gradient(x)
            self.assertAlmostEqual(energy, dense(x), places=12)
            for i in range(len(x)):
                a, b = x.copy(), x.copy()
                a[i] -= 1e-6; b[i] += 1e-6
                self.assertAlmostEqual(gradient[i], (dense(b)-dense(a))/2e-6, places=8)
            energies.append(energy)
            del _REGISTRY[id(k)]
        self.assertNotAlmostEqual(*energies, places=6)

    def test_complex_gates_and_y_terms_against_dense(self):
        spec=dict(n=3,initial='+',count=4,ops=[('ry',0,0),('rz',1,1),('cx',0,1),('rx',2,2),('cx',1,2),('ry',1,3)])
        words=['YXI','ZZI','III','XXZ'];coeffs=[.3,-.7,.4,.2]
        p=ReverseMPS(spec,words,coeffs,device='cpu');x=np.array([.2,-.4,.8,.1])
        e,g=p.value_gradient(x)
        # Independent dense circuit reference (q0 is leftmost tensor factor).
        pauli={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
        def dense(v):
            state=np.ones(8,dtype=complex)/np.sqrt(8)
            for gate,q,i in spec['ops']:
                if gate=='cx':
                    out=np.zeros(8,dtype=complex)
                    for j in range(8):out[j^((1<<(2-i)) if (j>>(2-q))&1 else 0)]=state[j]
                    state=out
                else:
                    P=pauli[gate[1].upper()];m=np.cos(v[i]/2)*np.eye(2)-1j*np.sin(v[i]/2)*P
                    op=np.array([1])
                    for k in range(3):op=np.kron(op,m if k==q else np.eye(2))
                    state=op@state
            h=np.zeros((8,8),complex)
            for w,c in zip(words,coeffs):
                op=np.array([1])
                for char in w:op=np.kron(op,pauli[char])
                h+=c*op
            return float(np.vdot(state,h@state).real)
        self.assertAlmostEqual(e,dense(x),places=12)
        for i in range(4):
            a=x.copy();b=x.copy();a[i]-=1e-6;b[i]+=1e-6
            self.assertAlmostEqual(g[i],(dense(b)-dense(a))/2e-6,places=8)
        # Chunk changes must not change energy or gradient.
        p.chunk=1;e1,g1=p.value_gradient(x)
        self.assertAlmostEqual(e,e1,places=12);np.testing.assert_allclose(g,g1,atol=1e-12)

    def test_unsupported_circuits_and_objectives_stay_finite_difference(self):
        k=object();register(k,3,9,('ry',),'linear',None,[],[],[])
        self.assertNotIn(id(k),_REGISTRY)
        self.assertIsNone(for_objective(lambda x:sum(x),'auto'))

    def test_adapter_uses_reverse_and_validates_energy(self):
        import lbfgs_optimizer as go
        class Problem:
            def value_gradient(self,x):return float(np.dot(x,x)),2*np.asarray(x)
        class BadProblem:
            def value_gradient(self,x):return 1e10,np.ones_like(x)
        for problem in (Problem(),BadProblem()):
            with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL':'1e-20','ANALYTIC_MPS_LBFGS_METRICS':str(Path(d)/'calls.jsonl')}),patch('reverse_mps.for_objective',return_value=problem):
                result=go.minimize(lambda x:float(np.dot(x,x)),[.7,-.2],options={'maxiter':6})
                self.assertLess(result.fun,1e-10)
                text=(Path(d)/'calls.jsonl').read_text()
                self.assertIn('reverse_fallback' if isinstance(problem,BadProblem) else 'reverse_mps',text)

    def test_required_mode_never_falls_back(self):
        import lbfgs_optimizer as go
        class BadProblem:
            def value_gradient(self,x):return 1e10,np.ones_like(x)
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
                'ANALYTIC_MPS_LBFGS_REVERSE_MODE':'required',
                'ANALYTIC_MPS_LBFGS_METRICS':str(Path(d)/'calls.jsonl')}):
            with patch('reverse_mps.for_objective',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'required but unavailable'):
                    go.minimize(lambda x:float(np.dot(x,x)),[.7,-.2])
            with patch('reverse_mps.for_objective',return_value=BadProblem()):
                with self.assertRaisesRegex(RuntimeError,'Finite differences are disabled'):
                    go.minimize(lambda x:float(np.dot(x,x)),[.7,-.2])

if __name__=='__main__':unittest.main()
