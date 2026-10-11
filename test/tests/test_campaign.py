import ast
from contextlib import nullcontext
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from optimization_trace import atomic
from job_state import reserve_case
from run_case import case_environment
from run_campaign import select_jobs


class CampaignTests(unittest.TestCase):
    def test_complete_matrix_and_templates(self):
        jobs=json.loads((ROOT/'campaign.json').read_text())['jobs']
        proteins=json.loads((ROOT/'protein_catalog.json').read_text())
        self.assertEqual(len(proteins),55)
        expected=set(itertools.product([p['protein_id'] for p in proteins],(-1,1000),('cobyla','lbfgs','spsa'),('linear','reverse_linear')))
        self.assertEqual(len(jobs),660)
        self.assertEqual({(j['protein_id'],j['shots'],j['optimizer'],j['entanglement']) for j in jobs},expected)
        self.assertEqual(len({j['key'] for j in jobs}),660)
        for job in jobs:
            self.assertEqual(job['method'],'independent-vqe')
            self.assertEqual(json.loads((ROOT/'jobs'/job['key']/'job.json').read_text()),{k:v for k,v in job.items() if k!='key'})
        selected=select_jobs(SimpleNamespace(proteins='3ckz',shots='-1,1000',optimizers=None,entanglements=None))
        self.assertEqual(len(selected),12)

    def test_settings_override_inherited_truncation_and_stopping(self):
        with patch.dict(os.environ,{'CUDAQ_MPS_MAX_BOND':'2','VQE_SHOTS':'12','ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL':'.01'}):
            for shots in (-1,1000):
                env=case_environment(dict(shots=shots,optimizer='lbfgs',entanglement='reverse_linear'),Path('/tmp/case'),1)
                self.assertNotIn('CUDAQ_MPS_MAX_BOND',env)
                self.assertEqual(env['CUDAQ_MPS_ABS_CUTOFF'],'0')
                self.assertEqual(env['CUDAQ_MPS_RELATIVE_CUTOFF'],'0')
                self.assertEqual(env['ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL'],'1')
                self.assertEqual(env['VQE_SHOTS'],str(shots))
                self.assertEqual(env['ANALYTIC_MPS_LBFGS_REVERSE_MODE'],'required' if shots==-1 else 'hybrid')

    def test_completion_lock_and_partial_restart(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)
            lock,reason=reserve_case(case);self.assertIsNone(reason)
            other,reason=reserve_case(case);self.assertIsNone(other);self.assertEqual(reason,'running')
            lock.close()
            atomic(case/'job_status.json',{'status':'failed'})
            lock,reason=reserve_case(case);self.assertIsNone(reason);lock.close()
            atomic(case/'job_status.json',{'status':'complete'})
            lock,reason=reserve_case(case);self.assertIsNone(lock);self.assertEqual(reason,'complete')

    def test_probe_isolation_best_parameters_and_completed_local_reuse(self):
        path=ROOT/'tools/independent_campaign.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
        cls.bases=[]
        fake=SimpleNamespace(probe=False,calls=0)
        fake.is_gradient_probe=lambda:fake.probe
        def minimize(fun,x,options):
            fake.calls+=1
            fun(np.array([2.]))
            fake.probe=True;fun(np.array([-100.]));fake.probe=False
            fun(np.array([1.]))
            fun(np.array([3.]))
            return SimpleNamespace(message='test stop',nit=2)
        fake.minimize=minimize
        scope=dict(hashlib=hashlib,json=json,os=os,Path=Path,SimpleNamespace=SimpleNamespace,np=np,optimizer=fake,atomic=atomic)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'ANALYTIC_MPS_TRACE_DIR':d}):
            solver=scope['CampaignVQE']()
            solver._evaluate_energy=lambda x,**kwargs:float(x[0])
            solver._timed_optimizer=lambda *args:nullcontext()
            model=dict(qubits=[0],kernel=None,hamiltonian=None,width=2)
            result=solver._minimize_model([0.],model,'local',100)
            self.assertEqual(result['energy'],1.)
            np.testing.assert_array_equal(result['params'],[1.])
            state=json.loads(next((Path(d)/'solves').glob('*.json')).read_text())
            self.assertEqual(state['latest_parameters'],[3.])
            self.assertEqual(state['best_parameters'],[1.])
            solver._minimize_model([0.],model,'local',100)
            self.assertEqual(fake.calls,1)
            solver._minimize_model([0.],dict(model,qubits=[1]),'local',100)
            self.assertEqual(fake.calls,2)
            self.assertEqual(len(list((Path(d)/'solves').glob('*.json'))),2)


if __name__=='__main__':unittest.main()
