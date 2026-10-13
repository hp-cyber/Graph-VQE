"""Original graph partition/circuit with common accelerated optimizers and restart."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from Qiskit_VQE.independent_vqe import IndependentSubspaceVQE
import campaign_optimizer as optimizer
from optimization_trace import atomic


class CampaignVQE(IndependentSubspaceVQE):
    def _minimize_model(self, params, model, phase, maxiter, qpu_id=0):
        # One namespace per local block/global solve, even for identical local Hs.
        key=phase+'_'+'_'.join(map(str,model['qubits']))
        path=Path(os.environ['ANALYTIC_MPS_TRACE_DIR'])/'solves'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
        self.checkpoint_manager=SimpleNamespace(checkpoint_path=str(path))
        state=json.loads(path.read_text()) if path.exists() else dict(
            phase=phase,qubits=model['qubits'],complete=False,best_energy=None,
            best_parameters=np.asarray(params).tolist(),global_candidates=[],
            global_ansatz='nearest_neighbor_su2_v1')
        if state.get('global_ansatz')!='nearest_neighbor_su2_v1':
            raise ValueError('Checkpoint predates the nearest-neighbor global ansatz; start a fresh case')
        if state['phase']!=phase or state['qubits']!=model['qubits']:
            raise ValueError('Checkpoint does not match partition')
        if phase=='global':
            for energy,x in state['global_candidates']:
                self._remember_global(energy,np.asarray(x))
        if state['complete']:
            print(f'[resume] Reusing completed {phase} solve for {model["qubits"]}',flush=True)
            result=dict(state['result']);result['params']=np.asarray(result['params'])
            return result

        def cost(x):
            probe=optimizer.is_gradient_probe()
            energy=self._evaluate_energy(x,qpu_id=qpu_id,kernel=model['kernel'],hamiltonian=model['hamiltonian'],
                                         evaluation_kind='gradient_probe' if probe else 'energy')
            if not probe:
                if state['best_energy'] is None or energy<state['best_energy']:
                    state.update(best_energy=float(energy),best_parameters=np.asarray(x).tolist())
                state.update(latest_parameters=np.asarray(x).tolist(),latest_energy=float(energy))
                if phase=='global':
                    self._remember_global(energy,np.asarray(x))
                    state['global_candidates']=[(e,p.tolist()) for e,p in self.iteration_results]
                atomic(path,state)
            return energy
        cost.reverse_problem=(model['kernel'],model['hamiltonian'])
        initial=np.asarray(state.get('latest_parameters',params),dtype=float)
        if maxiter and not model.get('constant',False):
            with self._timed_optimizer(phase,model['qubits'],qpu_id):
                result=optimizer.minimize(cost,initial,options={'maxiter':maxiter})
            reason=str(result.message)
            iterations=int(getattr(result,'nit',0))
        else:
            with self._timed_optimizer(phase,model['qubits'],qpu_id):cost(initial)
            reason='constant_local_objective' if model.get('constant',False) else 'iteration_limit'
            iterations=0
        result=dict(energy=state['best_energy'],params=state['best_parameters'],qubits=model['qubits'],
                    circuit_qubits=model['width'],accepted_iterations=iterations,stop_reason=reason)
        state.update(complete=True,result=result);atomic(path,state)
        return dict(result,params=np.asarray(result['params']))
