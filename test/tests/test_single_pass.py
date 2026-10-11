"""CPU-only tests executing production control flow with simulated energy calls."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import time
import unittest
from unittest.mock import Mock
import numpy as np


def production_methods():
    path = Path(__file__).resolve().parents[1]/'sources/independent-vqe/Qiskit_VQE/independent_vqe.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_run_vqe', '_remember_global', '_minimize_model')]
    module = ast.Module(body=[ast.ClassDef(name='Solver', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    namespace = dict(np=np, time=time, ThreadPoolExecutor=ThreadPoolExecutor,
                     configure_gpu_groups=Mock(), configure_full_bond=Mock(return_value=8))
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['Solver'], namespace


class SinglePassTests(unittest.TestCase):
    def test_one_solve_per_partition_then_one_global_using_best_parameters(self):
        for workers in (1, 2):
            for reason in ('energy_improvement_below_threshold', 'solver_converged', 'iteration_limit'):
                with self.subTest(workers=workers, reason=reason):
                    Solver, _ = production_methods();solver = Solver()
                    solver.seed=0;solver.num_params=6;solver.local_parameter_count=4
                    solver.num_parallel_workers=workers;solver.blocks=[[0], [1]]
                    solver.cross_terms=[('ZZ', 1)];solver.partition_summary={}
                    solver.num_gpus_detected=1;solver.num_qubits=2
                    solver.kernel=object();solver.hamiltonian=object()
                    solver.global_refinement=True;solver.max_epochs=1000000
                    solver.block_maxiter=100;solver.global_maxiter=100
                    solver._evaluation_seconds=0
                    solver.local_models=[dict(start=0,stop=2,width=2,kernel=Mock(),qubits=[0]),
                                         dict(start=2,stop=4,width=2,kernel=Mock(),qubits=[1])]
                    solver._timed_optimizer=lambda *a: nullcontext()
                    completed=[];global_calls=[]
                    expected=np.array([.1,.2,.3,.4,0.,0.])
                    def energy(x):
                        self.assertCountEqual(completed, [0,1])
                        np.testing.assert_allclose(x, expected)
                        return 20.
                    solver._evaluate_energy=energy
                    def optimize(x, model, phase, maxiter, qpu_id=0):
                        if phase=='local':
                            i=model['qubits'][0];completed.append(i)
                            best=expected[2*i:2*i+2].copy()
                            return dict(params=best,energy=float(i))
                        global_calls.append(x.copy())
                        self.assertEqual(len(global_calls),1, 'Global optimizer repeated')
                        self.assertCountEqual(completed,[0,1])
                        np.testing.assert_allclose(x,expected)
                        best=x+.01
                        solver._remember_global(5.,best)
                        return dict(params=best,energy=5.,stop_reason=reason)
                    solver._minimize_model=optimize
                    _,best,_,_=solver._run_vqe(np.ones(6))
                    self.assertEqual(len(global_calls),1)
                    self.assertEqual(len(solver.global_results),1)
                    self.assertEqual(len(solver.epoch_summaries),1)
                    np.testing.assert_allclose(solver.product_params,expected)
                    np.testing.assert_allclose(best,expected+.01)



if __name__=='__main__':
    unittest.main()
