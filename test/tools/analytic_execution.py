"""No-shot MPS expectations and analytic equivalents of sampled Z moments.

Never extract a full statevector. Forging superposition circuits and pVQE local
moments retain their original coefficients, partitions and reconstruction math.
"""
import json
from functools import lru_cache
import os
from pathlib import Path
import threading
import time
import numpy as np

_log_lock=threading.Lock()

@lru_cache(maxsize=256)
def moment_observable(supports):
    """Reuse immutable Pauli observables, never energy results or samples."""
    import cudaq
    ops=[]
    for support in supports:
        op=cudaq.spin.z(support[0])
        for q in support[1:]:op=op*cudaq.spin.z(q)
        ops.append(op)
    h=ops[0]
    for op in ops[1:]:h=h+op
    return h,ops


class AnalyticMoments:
    def __init__(self,kernel,parameters,supports):
        import cudaq
        self.values={():1.0}
        supports=sorted({tuple(sorted(s)) for s in supports if s})
        if not supports:return
        h,ops=moment_observable(tuple(supports))
        values=np.asarray(parameters,dtype=float).reshape(-1).tolist()
        result=cudaq.observe(kernel,h,*values,shots_count=-1)
        for support,op in zip(supports,ops):self.values[support]=float(result.expectation(op))
    def moment(self,support):
        return self.values[tuple(sorted(support))]


def install(method):
    import cudaq
    from circuit_tape import native
    if (os.environ.get('ANALYTIC_MPS_OPTIMIZER') == 'lbfgs' and
            os.environ.get('ANALYTIC_MPS_LBFGS_REVERSE_MODE', 'auto') != 'off'):
        from circuit_tape import install as install_tape
        install_tape(cudaq)
    native_observe=cudaq.observe
    native_async=cudaq.observe_async
    def record(kind,started,args):
        path=Path(os.environ.get('ANALYTIC_MPS_BACKEND_METRICS','backend_evaluations.jsonl'))
        data=dict(operation=kind,elapsed_seconds=time.perf_counter()-started,shots=-1,
                  physical_shots=0,target='tensornet-mps-mqpu-shots',
                  mps_max_bond=os.environ.get('CUDAQ_MPS_MAX_BOND'),
                  abs_cutoff=os.environ.get('CUDAQ_MPS_ABS_CUTOFF'),
                  relative_cutoff=os.environ.get('CUDAQ_MPS_RELATIVE_CUTOFF'))
        parameters=np.asarray(args[2:],dtype=float).reshape(-1).tolist()
        data.update(parameters=parameters,parameter_count=len(parameters))
        with _log_lock,path.open('a') as f:f.write(json.dumps(data)+'\n')
    def check(kwargs):
        if kwargs.get('shots_count',-1)!=-1:raise RuntimeError('Finite-shot observe forbidden in analytic campaign')
        if kwargs.get('noise_model') is not None:raise RuntimeError('Noise model forbidden in analytic campaign')
        kwargs['shots_count']=-1
    def observe(*args,**kwargs):
        check(kwargs);start=time.perf_counter();out=native_observe(native(args[0]),*args[1:],**kwargs);record('observe',start,args);return out
    class AsyncResult:
        def __init__(self,result,start,args):self.result=result;self.start=start;self.args=args;self.logged=False
        def get(self):
            out=self.result.get()
            if not self.logged:record('observe_async_until_get',self.start,self.args);self.logged=True
            return out
    def observe_async(*args,**kwargs):
        check(kwargs);start=time.perf_counter();return AsyncResult(native_async(native(args[0]),*args[1:],**kwargs),start,args)
    def forbidden(*args,**kwargs):
        raise RuntimeError('Sampling/statevector readout forbidden: this campaign records analytic optimization only')
    cudaq.observe=observe;cudaq.observe_async=observe_async
    cudaq.sample=forbidden;cudaq.sample_async=forbidden;cudaq.get_state=forbidden
    if method in ('pvqe','classically-split-vqe','entanglement-forging-vqe','edef-adapt-vqe'):
        import Qiskit_VQE.published_baselines as module
        old_moment=module.z_moment
        module.z_moment=lambda values,support: values.moment(support) if isinstance(values,AnalyticMoments) else old_moment(values,support)
        def blocks(self,parameters):
            return [AnalyticMoments(kernel,parameters[self.offsets[i]:self.offsets[i+1]],
                    [[local for local,q in enumerate(block) if q in term.qubits] for term in self.terms])
                    for i,(kernel,block) in enumerate(zip(self.kernels,self.blocks))]
        module.PartitionedVQE._sample_blocks=blocks
        def family(self,side,theta):
            from itertools import combinations
            bases=self.basis_a if side=='a' else self.basis_b
            block=self.partition_a if side=='a' else self.partition_b
            locations={q:i for i,q in enumerate(block)}
            supports=[[locations[q] for q in term.qubits if q in locations] for term in self.terms]
            diagonal=[AnalyticMoments(self._kernel(side,('basis',bits)),theta,supports) for bits in bases]
            pairs={}
            for left,right in combinations(range(self.schmidt_rank),2):
                for phase in range(4):
                    pairs[(left,right,phase)]=AnalyticMoments(self._kernel(side,('pair',bases[left],bases[right],phase)),theta,supports)
            return diagonal,pairs
        module.EntanglementForgingVQE._sample_family=family
        for cls in (module.PartitionedVQE,module.EntanglementForgingVQE,module.EDEFAdaptVQE,module.ExtendedClassicallySplitVQE):
            old=cls.algorithm_metadata
            def metadata(self,_old=old):
                data=_old(self);data.update(shots=-1,shot_noise_enabled=False,readout_performed=False,
                    readout='disabled in analytic optimization campaign',
                    moment_evaluation='analytic MPS expectations; original reconstruction')
                for key in list(data):
                    if key.startswith('aggregate_') and key.endswith('_shots'):data[key]=0
                return data
            cls.algorithm_metadata=metadata
    if method=='cvar-vqe':
        from Qiskit_VQE.cvar_vqe import CVaRVQE
        def exact(self,parameters):
            if self.alpha!=1.0:raise ValueError('Analytic campaign preserves source alpha=1; general exact CVaR is not implemented')
            return self.observe_energy(parameters)
        CVaRVQE._cvar_energy=exact
    if method=='eqc-vqe':
        from Qiskit_VQE.eqc_vqe import EQCVQE
        # Each campaign job has one physical GPU/client. With no sampling and
        # no device-noise channel, its weighted ensemble is exactly this energy.
        def start_clients(self):
            if len(self.client_gpus)!=1:raise RuntimeError('Analytic EQC campaign expects one assigned GPU/client')
        EQCVQE._start_clients=start_clients
        EQCVQE._close_clients=lambda self:None
        def energy(self,parameters):
            start=time.perf_counter()
            value=float(cudaq.observe(self.kernel,self.hamiltonian,*np.asarray(parameters).tolist(),shots_count=-1).expectation())
            self._last_client_energies=[value];self._last_client_elapsed=[time.perf_counter()-start]
            return value
        EQCVQE._energy=energy
