"""First-order SPSA for the original VQE scalar objectives.

Standard Rademacher simultaneous perturbations, power-law gain schedules, and
optional automatic gain calibration. One update uses two perturbation probes
plus one recorded objective at the updated parameters. Calibration and probes
cannot modify best-state/checkpoint/plateau history in the copied solvers.
"""
from contextvars import ContextVar
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from scipy.optimize import OptimizeResult

_probe=ContextVar('spsa_gradient_probe',default=False)
_log_lock=threading.Lock()


def is_gradient_probe():
    return _probe.get()


@contextmanager
def gradient_probe():
    token=_probe.set(True)
    try:yield
    finally:_probe.reset(token)


def _log(record):
    path=Path(os.environ.get('ANALYTIC_MPS_SPSA_METRICS','spsa_evaluations.jsonl'))
    path.parent.mkdir(parents=True,exist_ok=True)
    with _log_lock,path.open('a') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n')


def settings():
    """Explicit, reproducible first-order SPSA hyperparameters."""
    values=dict(perturbation=float(os.environ.get('ANALYTIC_MPS_SPSA_PERTURBATION','0.2')),
        learning_rate=None if os.environ.get('ANALYTIC_MPS_SPSA_LEARNING_RATE','auto')=='auto' else float(os.environ['ANALYTIC_MPS_SPSA_LEARNING_RATE']),
        alpha=float(os.environ.get('ANALYTIC_MPS_SPSA_ALPHA','0.602')),
        gamma=float(os.environ.get('ANALYTIC_MPS_SPSA_GAMMA','0.101')),
        stability=float(os.environ.get('ANALYTIC_MPS_SPSA_STABILITY','0')),
        calibration_pairs=int(os.environ.get('ANALYTIC_MPS_SPSA_CALIBRATION_PAIRS','5')),
        target_update=float(os.environ.get('ANALYTIC_MPS_SPSA_TARGET_UPDATE',str(2*np.pi/10))),
        seed=int(os.environ.get('ANALYTIC_MPS_SPSA_SEED','0')))
    for key in ('perturbation','alpha','gamma','target_update'):
        if not np.isfinite(values[key]) or values[key]<=0:raise ValueError(f'Invalid SPSA {key}')
    if not np.isfinite(values['stability']) or values['stability']<0:raise ValueError('Invalid SPSA stability')
    if values['learning_rate'] is not None and (not np.isfinite(values['learning_rate']) or values['learning_rate']<=0):raise ValueError('Invalid SPSA learning rate')
    if values['calibration_pairs']<1:raise ValueError('SPSA calibration_pairs must be positive')
    return values


def perturbation_gradient(fun,x,c,rng):
    """Two scalar calls estimate the entire gradient, independently of len(x)."""
    delta=rng.choice(np.array([-1.,1.]),size=x.size)
    plus=float(fun(x+c*delta));minus=float(fun(x-c*delta))
    return ((plus-minus)/(2*c))*delta


def minimize(fun,x0,args=(),method=None,jac=None,hess=None,hessp=None,
             bounds=None,constraints=(),tol=None,callback=None,options=None):
    if constraints:raise ValueError('SPSA adapter does not implement general constraints')
    from optimization_trace import active
    journal=active.get()
    restored=journal.state.get('spsa') if journal is not None and journal.resumed else None
    config=settings();options=options or {}
    if restored and restored['settings'] != config: raise ValueError('SPSA resume settings changed')
    cap=min(int(os.environ.get('ANALYTIC_MPS_SPSA_MAX_ITER','1000000')),int(options.get('maxiter',1000000)))
    if cap<1:raise ValueError('Positive SPSA iteration cap required')
    if restored and restored.get('pending'):
        pending=restored['pending']
        if np.array_equal(np.asarray(journal.state['x']),np.asarray(pending['x'])):
            restored=dict(restored, **pending, energy=journal.state['energy'])
            if restored['energy']<restored['best_energy']:
                restored['best_energy']=restored['energy'];restored['best_x']=restored['x']
        else:
            # The pending objective never completed: replay this update.
            journal.state['x']=restored['x']
        x0=restored['x']
    x=np.asarray(x0,dtype=float).copy()
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Expected finite one-dimensional initial parameters')
    if bounds is None:
        lower=np.full(x.size,-np.inf);upper=np.full(x.size,np.inf)
    elif hasattr(bounds,'lb'):
        lower=np.broadcast_to(bounds.lb,x.shape);upper=np.broadcast_to(bounds.ub,x.shape)
    else:
        lower=np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        upper=np.array([np.inf if b[1] is None else b[1] for b in bounds])
    if lower.shape!=x.shape or upper.shape!=x.shape or np.any(lower>upper):raise ValueError('Invalid bounds')
    x=np.clip(x,lower,upper)
    # Local RNG: independent of NumPy global state and other block threads.
    rng=np.random.default_rng(config['seed'])
    call_id=uuid.uuid4().hex
    header=dict(optimizer='SPSA',optimizer_call=call_id,objective=getattr(fun,'__qualname__',repr(fun)))
    counts=dict(objective=0,gradient_probe=0,calibration_probe=0,gradient=0,accepted_iteration=0)
    started=time.perf_counter();status='interrupted_or_solver_early_stop';nit=0
    _log(dict(header,event='start',parameters=x.size,maxiter=cap,settings=config))
    print(f'[SPSA] {header["objective"]}; parameters={x.size}; max updates={cap}; calibration pairs={config["calibration_pairs"] if config["learning_rate"] is None else 0}',flush=True)

    def evaluate(point,kind='objective'):
        t=time.perf_counter()
        if kind!='objective':
            import work_trace as work
            with work.scope(probe_kind=kind), gradient_probe():value=float(fun(point,*args))
        else:value=float(fun(point,*args))
        if not np.isfinite(value):raise ValueError(f'Nonfinite SPSA {kind}: {value}')
        counts[kind]+=1
        _log(dict(header,event=kind,index=counts[kind],energy=value,seconds=time.perf_counter()-t))
        return value

    try:
        fx=evaluate(x);best_x=x.copy();best_f=fx
        first=1
        if restored:
            x=np.asarray(restored['x'],dtype=float);fx=float(restored['energy'])
            best_x=np.asarray(restored['best_x']);best_f=float(restored['best_energy'])
            rng.bit_generator.state=restored['rng_state']
            first=int(restored['iteration'])+1
        if not x.size:
            status='No variational parameters'
            return OptimizeResult(x=x,fun=fx,nit=0,nfev=1,njev=0,success=True,message=status)
        c=config['perturbation'];a=restored['learning_rate'] if restored else config['learning_rate']
        if a is None:
            # Match standard SPSA calibration: target initial component update
            # divided by average magnitude of directional finite differences.
            magnitudes=[]
            for _ in range(config['calibration_pairs']):
                g=perturbation_gradient(lambda p:evaluate(p,'calibration_probe'),x,c,rng)
                magnitudes.append(float(np.mean(np.abs(g))))
            magnitude=float(np.mean(magnitudes))
            a=config['target_update']*(config['stability']+1)**config['alpha']/magnitude if magnitude>1e-12 else 1.0
            _log(dict(header,event='calibration',learning_rate=a,mean_directional_magnitude=magnitude,pairs=config['calibration_pairs']))
        if journal is not None and not restored:
            journal.save(spsa=dict(iteration=0,x=x.tolist(),energy=fx,best_x=best_x.tolist(),
                best_energy=best_f,learning_rate=a,settings=config,rng_state=rng.bit_generator.state,counts=counts))
        print(f'[SPSA] a={a:.6g}, c={c}, alpha={config["alpha"]}, gamma={config["gamma"]}',flush=True)
        for k in range(first,first+cap):
            previous_fx=fx
            ak=a/(k+config['stability'])**config['alpha']
            ck=c/k**config['gamma']
            t=time.perf_counter()
            import work_trace as work
            with work.scope(role='gradient'):
                with work.Span('gradient_evaluation', parameters=x.tolist(), parameter_count=x.size) as event:
                    g=perturbation_gradient(lambda p:evaluate(p,'gradient_probe'),x,ck,rng)
                    event.result['gradient']=g.tolist()
            counts['gradient']+=1
            _log(dict(header,event='gradient',index=k,norm=float(np.linalg.norm(g)),seconds=time.perf_counter()-t,learning_rate=ak,perturbation=ck))
            candidate=np.clip(x-ak*g,lower,upper)
            if journal is not None:
                saved=dict(journal.state['spsa'])
                saved['pending']=dict(x=candidate.tolist(),iteration=k,rng_state=rng.bit_generator.state)
                journal.save(spsa=saved)
            fx=evaluate(candidate)
            x=candidate;nit=k;counts['accepted_iteration']+=1
            if fx<best_f:best_x=x.copy();best_f=fx
            # Save a diagnostic optimizer checkpoint after each recorded update.
            # Source solvers own restart/topology checkpoints; this file does
            # not claim exact recovery of an interrupted outer VQE solve.
            state_dir=Path(os.environ.get('ANALYTIC_MPS_SPSA_STATE_DIR','spsa_states'))
            state_dir.mkdir(parents=True,exist_ok=True)
            state=dict(header,iteration=k,x=x.tolist(),energy=fx,best_x=best_x.tolist(),best_energy=best_f,
                       learning_rate=a,settings=config,rng_state=rng.bit_generator.state,counts=counts)
            final=state_dir/(call_id+'.json');tmp=final.with_suffix('.tmp')
            tmp.write_text(json.dumps(state)+'\n');os.replace(tmp,final)
            if journal is not None: journal.save(spsa=state, x=x.tolist(), energy=fx)
            if callback is not None:callback(x)
            threshold=float(os.environ.get('ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL','1'))
            if abs(fx-previous_fx)<threshold:
                status=f'Immediate energy stopping: |delta E| < {threshold}; no patience/window'
                _log(dict(header,event='immediate_energy_stop',absolute_energy_change=abs(fx-previous_fx),tolerance=threshold,patience=0,window=0))
                break
        if status=='interrupted_or_solver_early_stop':status='SPSA iteration cap reached'
        return OptimizeResult(x=best_x,fun=best_f,nit=nit,nfev=sum(counts[v] for v in ('objective','gradient_probe','calibration_probe')),
                              njev=counts['gradient'],success=status.startswith('Immediate energy'),message=status)
    finally:
        _log(dict(header,event='finish',status=status,counts=counts,seconds=time.perf_counter()-started))
        print(f'[SPSA] {status}; counts={counts}',flush=True)
