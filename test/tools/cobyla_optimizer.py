"""COBYLA on the unchanged scalar objectives, with an explicit call budget."""
import json
import os
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from scipy.optimize import minimize as scipy_minimize, OptimizeResult

_lock=threading.Lock()

def is_gradient_probe():
    return False

def _log(data):
    path=Path(os.environ.get('ANALYTIC_MPS_METRICS','cobyla_evaluations.jsonl'))
    with _lock,path.open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')

def minimize(fun,x0,args=(),method=None,jac=None,hess=None,hessp=None,
             bounds=None,constraints=(),tol=None,callback=None,options=None):
    options=dict(options or {})
    cap=min(int(os.environ.get('ANALYTIC_MPS_MAX_ITER','1000000')),int(options.get('maxiter',1000000)))
    if cap<1:raise ValueError('Positive objective evaluation cap required')
    x0=np.asarray(x0,dtype=float);best=[float('inf'),x0.copy()];count=0
    header=dict(optimizer='COBYLA',optimizer_call=uuid.uuid4().hex,objective=getattr(fun,'__qualname__',repr(fun)))
    started=time.perf_counter();status='interrupted_or_source_early_stop'
    immediate=os.environ.get('ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL','1')
    immediate=None if immediate is None else float(immediate)
    if immediate is not None and (not np.isfinite(immediate) or immediate<=0):
        raise ValueError('Immediate energy tolerance must be finite and positive')
    previous=None
    class ImmediateEnergyStop(Exception):pass
    class BudgetReached(Exception):pass
    def evaluate(x):
        nonlocal count, previous
        if count>=cap:raise BudgetReached()
        begin=time.perf_counter();value=float(fun(x,*args));count+=1
        if not np.isfinite(value):raise ValueError('Nonfinite objective')
        if value<best[0]:best[:]=[value,np.asarray(x).copy()]
        _log(dict(header,event='objective',index=count,energy=value,seconds=time.perf_counter()-begin))
        delta=None if previous is None else abs(value-previous)
        previous=value
        if immediate is not None and delta is not None and delta<immediate:
            _log(dict(header,event='immediate_energy_stop',index=count,
                      absolute_energy_change=delta,tolerance=immediate,
                      patience=0,window=0,best_energy=best[0]))
            raise ImmediateEnergyStop()
        return value
    _log(dict(header,event='start',parameters=x0.size,max_evaluations=cap,shots=int(os.environ.get('VQE_SHOTS','-1')),immediate_energy_tolerance=immediate))
    try:
        if x0.size==0:
            value=evaluate(x0);status='No variational parameters'
            return OptimizeResult(x=x0,fun=value,nfev=count,success=True,message=status)
        # PRIMA needs n+2 internal interpolation calls; the wrapper enforces the
        # actual requested budget even for deliberately tiny smoke-test caps.
        options['maxiter']=max(cap,x0.size+2)
        try:
            result=scipy_minimize(evaluate,x0,method='COBYLA',bounds=bounds,
                constraints=constraints,tol=tol,callback=callback,options=options)
            status=str(result.message)
            result.x=best[1];result.fun=best[0];result.nfev=count
            return result
        except ImmediateEnergyStop:
            status=f'Immediate energy stopping: consecutive |delta E| < {immediate:g} (no patience/window)'
            return OptimizeResult(x=best[1],fun=best[0],nfev=count,success=True,message=status)
        except BudgetReached:
            status='Recorded objective evaluation cap reached'
            return OptimizeResult(x=best[1],fun=best[0],nfev=count,success=False,message=status)
    finally:
        _log(dict(header,event='finish',status=status,counts=dict(objective=count),seconds=time.perf_counter()-started))
