"""1000-shot counterpart of analytic_execution; no added hardware error model."""
import json
import os
from pathlib import Path
import threading
import time
import numpy as np

_lock=threading.Lock()


def install(method):
    import cudaq
    from circuit_tape import native
    shots=int(os.environ['VQE_SHOTS'])
    if shots != 1000: raise ValueError('Matched sampled suite requires 1000 shots')
    if os.environ.get('ANALYTIC_MPS_OPTIMIZER') == 'lbfgs':
        from circuit_tape import install as install_tape
        install_tape(cudaq)
    def log(operation, start, args):
        row=dict(operation=operation, elapsed_seconds=time.perf_counter()-start,
                 shots=shots, physical_shots_per_circuit_or_term=shots,
                 target='tensornet-mps-mqpu-shots', added_noise_model=False)
        parameters=np.asarray(args[1:] if operation.startswith('sample') else args[2:],dtype=float).reshape(-1).tolist()
        row.update(parameters=parameters,parameter_count=len(parameters))
        with _lock, Path(os.environ['ANALYTIC_MPS_BACKEND_METRICS']).open('a') as f:
            f.write(json.dumps(row)+'\n')
    class Future:
        def __init__(self, result, start, name, args):
            self.result=result;self.start=start;self.name=name;self.args=args;self.logged=False
        def get(self):
            out=self.result.get()
            if not self.logged:log(self.name,self.start,self.args);self.logged=True
            return out
    def wrap(name):
        original=getattr(cudaq,name)
        def call(*args, **kwargs):
            kwargs['shots_count']=shots
            start=time.perf_counter()
            result=original(native(args[0]),*args[1:],**kwargs)
            if name.endswith('_async'):return Future(result,start,name,args)
            log(name,start,args)
            return result
        setattr(cudaq,name,call)
    for name in ('observe','observe_async','sample','sample_async'):wrap(name)
    def forbidden(*args,**kwargs):raise RuntimeError('Dense state extraction is disabled; use sampled readout')
    cudaq.get_state=forbidden
    if method=='eqc-vqe':
        from Qiskit_VQE.eqc_vqe import EQCVQE
        def start(self):
            if len(self.client_gpus)!=1:raise RuntimeError('Matched EQC expects one client')
        EQCVQE._start_clients=start;EQCVQE._close_clients=lambda self:None
        def energy(self,parameters):
            begin=time.perf_counter()
            value=float(cudaq.observe(self.kernel,self.hamiltonian,*np.asarray(parameters).tolist(),shots_count=shots).expectation())
            self._last_client_energies=[value];self._last_client_elapsed=[time.perf_counter()-begin]
            return value
        EQCVQE._energy=energy
