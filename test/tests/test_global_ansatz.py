"""Production circuit construction and mapping without allocating a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]


class GlobalAnsatzTests(unittest.TestCase):
    def test_global_kernel_is_one_chain_with_layer_qubit_gate_order(self):
        source=ROOT/'sources/independent-vqe/Qiskit_VQE/independent_vqe.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_build_kernel')
        class Kernel:
            def __init__(self):self.ops=[]
            def qalloc(self,n):return list(range(n))
            def ry(self,p,q):self.ops.append(('ry',p,q))
            def rz(self,p,q):self.ops.append(('rz',p,q))
            def cx(self,a,b):self.ops.append(('cx',a,b))
        def make(*types):return Kernel(),*range(len(types))
        def pairs(n,layer,pattern):
            indices=range(n-1) if pattern=='linear' else range(n-2,-1,-1)
            return [(i,i+1) for i in indices]
        scope=dict(cudaq=SimpleNamespace(make_kernel=make),pairs_for_pattern=pairs)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
        for n in (7,97):
            for pattern in ('linear','reverse_linear'):
                solver=SimpleNamespace(reps=4,su2_gates=('ry','rz'),entanglement=pattern)
                kernel=scope['_build_kernel'](solver,[list(range(n))],n)
                rotations=[op for op in kernel.ops if op[0]!='cx']
                self.assertEqual(len(rotations),10*n)
                self.assertEqual([p for _,p,_ in rotations],list(range(10*n)))
                self.assertEqual([q for _,_,q in rotations],[q for _ in range(5) for q in range(n) for _ in range(2)])
                self.assertEqual([(a,b) for name,a,b in kernel.ops if name=='cx'],pairs(n,0,pattern)*4)

    def test_all_strategy_copies_match(self):
        source=(ROOT/'sources/independent-vqe/Qiskit_VQE/independent_vqe.py').read_bytes()
        for pattern in ('linear','reverse_linear'):
            path=ROOT.parent/f'VQE_Docking_cuda-q_independent_{pattern}/Qiskit_VQE/independent_vqe.py'
            self.assertEqual(path.read_bytes(),source)


if __name__=='__main__':unittest.main()
