import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import campaign_optimizer as dispatch
import lbfgs_optimizer as lb
import spsa_optimizer as sp

class OptimizerTests(unittest.TestCase):
    def test_first_change_below_one_stops_all_optimizers(self):
        for key in ('cobyla','lbfgs','spsa'):
            with self.subTest(optimizer=key),tempfile.TemporaryDirectory() as d:
                env=dict(ANALYTIC_MPS_TRACE_DIR=d,ANALYTIC_MPS_OPTIMIZER=key,
                    ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL='1',ANALYTIC_MPS_CENTRAL_STOP='1',
                    ANALYTIC_MPS_LBFGS_REVERSE_MODE='off',ANALYTIC_MPS_SPSA_LEARNING_RATE='.1',
                    ANALYTIC_MPS_METRICS=str(Path(d)/'cobyla.jsonl'),
                    ANALYTIC_MPS_LBFGS_METRICS=str(Path(d)/'lbfgs.jsonl'),
                    ANALYTIC_MPS_SPSA_METRICS=str(Path(d)/'spsa.jsonl'),
                    ANALYTIC_MPS_SPSA_STATE_DIR=str(Path(d)/'states'))
                with patch.dict(os.environ,env):
                    # The first change is less than one; no history window or
                    # additional iteration is needed, regardless of optimizer.
                    result=dispatch.minimize(lambda x:.1*float(x@x),[1.],options={'maxiter':100})
                self.assertIn('no patience/window',str(result.message))
                rows=[json.loads(s) for s in (Path(d)/'optimization_trace.jsonl').read_text().splitlines()]
                objectives=[r for r in rows if r['event']=='energy_evaluation' and r['kind']=='objective']
                self.assertEqual(len(objectives),2)

    def test_analytic_gradient_and_convergence(self):
        x=np.array([.4,-.7]);f=lambda x:(np.cos(x[0])+np.sin(x[1]))**2
        exact=2*(np.cos(x[0])+np.sin(x[1]))*np.array([-np.sin(x[0]),np.cos(x[1])])
        np.testing.assert_allclose(lb.central_gradient(f,x,1e-5),exact,atol=1e-8)
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'ANALYTIC_MPS_TRACE_DIR':d,'ANALYTIC_MPS_OPTIMIZER':'lbfgs','ANALYTIC_MPS_LBFGS_METRICS':str(Path(d)/'metrics.jsonl'),'ANALYTIC_MPS_LBFGS_MAX_ITER':'30'}):
            history=[]
            def energy(x):
                value=float(np.sum((x-[1.,-2.])**2))
                if not dispatch.is_gradient_probe():history.append(value)
                return value
            result=dispatch.minimize(energy,[4.,3.])
            self.assertLess(result.fun,1e-10)
            rows=[json.loads(s) for s in (Path(d)/'metrics.jsonl').read_text().splitlines()]
            counts=rows[-1]['counts'];self.assertEqual(counts['objective'],len(history))
            self.assertEqual(counts['gradient_probe'],4*counts['gradient'])

    def test_spsa_probes_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'ANALYTIC_MPS_TRACE_DIR':d,'ANALYTIC_MPS_OPTIMIZER':'spsa','ANALYTIC_MPS_SPSA_METRICS':str(Path(d)/'metrics.jsonl'),'ANALYTIC_MPS_SPSA_STATE_DIR':str(Path(d)/'states'),'ANALYTIC_MPS_SPSA_MAX_ITER':'20','ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL':'1e-30','ANALYTIC_MPS_SPSA_LEARNING_RATE':'0.1'}):
            history=[]
            def energy(x):
                value=float(np.sum((x-[1.,-2.])**2))
                if not dispatch.is_gradient_probe():history.append(value)
                return value
            result=dispatch.minimize(energy,[4.,3.])
            self.assertLess(result.fun,history[0]);self.assertEqual(len(history),21)
            rows=[json.loads(s) for s in (Path(d)/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['counts']['gradient_probe'],40)
            self.assertEqual(len(list((Path(d)/'states').glob('*.json'))),1)

    def test_probe_context_reset(self):
        for key in ('lbfgs','spsa'):
            with patch.dict(os.environ,{'ANALYTIC_MPS_OPTIMIZER':key}):
                with self.assertRaises(RuntimeError):
                    with dispatch.backend().gradient_probe():raise RuntimeError('stop')
                self.assertFalse(dispatch.is_gradient_probe())

if __name__=='__main__':unittest.main()
