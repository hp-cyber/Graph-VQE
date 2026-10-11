"""L-BFGS-B on each method's original scalar objective and its reverse derivative.

The production launcher requires reverse mode, including reconstruction chain
rules. Diagnostic auto/off modes retain finite differences. Verification probes
bypass optimization history but remain visible in the evaluation log.
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
from scipy.optimize import minimize as scipy_minimize, OptimizeResult

_probe = ContextVar('gradient_probe', default=False)
_log_lock = threading.Lock()


def is_gradient_probe():
    return _probe.get()


@contextmanager
def gradient_probe():
    token = _probe.set(True)
    try:
        yield
    finally:
        _probe.reset(token)


def _log(record):
    target = Path(os.environ.get('ANALYTIC_MPS_LBFGS_METRICS', 'gradient_evaluations.jsonl'))
    target.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, target.open('a') as out:
        out.write(json.dumps(record, allow_nan=False) + '\n')


def central_gradient(fun, x, step, bounds=None):
    """Central differences; one-sided differences at an explicit bound."""
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    for i in range(x.size):
        lo, hi = (None, None) if bounds is None else bounds[i]
        a, b = x.copy(), x.copy()
        a[i] = max(x[i] - step, -np.inf if lo is None else lo)
        b[i] = min(x[i] + step, np.inf if hi is None else hi)
        result[i] = 0.0 if a[i] == b[i] else (fun(b) - fun(a)) / (b[i] - a[i])
    return result


def minimize(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    if constraints:
        raise ValueError('L-BFGS-B cannot implement general constraints')
    options = options or {}
    cap = int(os.environ.get('ANALYTIC_MPS_LBFGS_MAX_ITER', '1000000'))
    cap = min(cap, int(options.get('maxiter', cap)))
    step = float(os.environ.get('ANALYTIC_MPS_LBFGS_FD_STEP', '1e-5'))
    if not np.isfinite(step) or step <= 0 or cap < 1:
        raise ValueError('Positive finite gradient step and iteration cap required')
    immediate = os.environ.get('ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL','1')
    immediate = None if immediate is None else float(immediate)
    if immediate is not None and (not np.isfinite(immediate) or immediate <= 0):
        raise ValueError('Immediate energy tolerance must be finite and positive')
    previous = None
    best = [float('inf'), np.asarray(x0, dtype=float).copy()]
    class ImmediateEnergyStop(Exception):
        pass
    if hasattr(bounds, 'lb'):
        bounds = list(zip(bounds.lb, bounds.ub))
    from reverse_mps import for_objective
    reverse_mode = os.environ.get('ANALYTIC_MPS_LBFGS_REVERSE_MODE','auto')
    try:
        reverse = for_objective(fun, reverse_mode) if not args else None
    except ImportError as exc:
        if reverse_mode in ('required','hybrid'): raise
        print(f'[gradient] reverse mode unavailable ({exc}); using finite differences',flush=True)
        reverse = None
    if reverse is None and reverse_mode in ('required','hybrid') and len(x0):
        raise RuntimeError(f'Reverse gradient is required but unavailable for {getattr(fun, "__qualname__", fun)}')
    reverse_checked = False
    if reverse is not None:
        print(f'[gradient] reverse derivative enabled: {getattr(reverse, "engine", "reverse_mps")}; objective evaluations retain their configured shots',flush=True)
    call_id = uuid.uuid4().hex
    count = dict(objective=0, gradient_probe=0, gradient=0, accepted_iteration=0)
    started = time.perf_counter()
    name = getattr(fun, '__qualname__', repr(fun))
    header = dict(optimizer_call=call_id, objective=name, optimizer='L-BFGS-B')
    _log(dict(header, event='start', parameters=len(x0), maxiter=cap, immediate_energy_tolerance=immediate,
              derivative=getattr(reverse, 'engine', 'reverse_mps') if reverse is not None else 'central finite differences of actual objective', reverse_mode=reverse_mode, step=step))
    print(f"[gradient] L-BFGS-B: {name}; parameters={len(x0)}; derivative={getattr(reverse, 'engine', 'reverse_mps') if reverse is not None else 'finite differences'}; cap={cap}", flush=True)

    def evaluate(x, probe=False):
        nonlocal previous
        kind = 'gradient_probe' if probe else 'objective'
        t = time.perf_counter()
        if probe:
            with gradient_probe():
                value = float(fun(x, *args))
        else:
            value = float(fun(x, *args))
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite {kind} energy: {value}')
        count[kind] += 1
        _log(dict(header, event=kind, index=count[kind], energy=value,
                  seconds=time.perf_counter()-t))
        if not probe:
            if value < best[0]:
                best[:] = [value, np.asarray(x, dtype=float).copy()]
            delta = None if previous is None else abs(value-previous)
            previous = value
            if immediate is not None and delta is not None and delta < immediate:
                _log(dict(header, event='immediate_energy_stop', index=count['objective'],
                          absolute_energy_change=delta, tolerance=immediate,
                          patience=0, window=0, best_energy=best[0]))
                raise ImmediateEnergyStop()
        return value

    def derivative(x):
        nonlocal reverse, reverse_checked
        count['gradient'] += 1
        t = time.perf_counter()
        if reverse is not None:
            analytic_energy,value = reverse.value_gradient(x)
            if reverse_mode == 'required' or (not reverse_checked and reverse_mode == 'auto'):
                observed = evaluate(x, True)
                if not np.isclose(analytic_energy, observed, rtol=1e-9, atol=1e-7):
                    if reverse_mode == 'required':
                        raise RuntimeError(f'Reverse energy {analytic_energy} differs from CUDA-Q objective {observed}; exact gradient may not match a truncated simulator state. Finite differences are disabled.')
                    _log(dict(header,event='reverse_fallback',reason='analytic energy mismatch',
                              reverse_energy=analytic_energy,backend_energy=observed))
                    reverse = None
                    return central_gradient(lambda p: evaluate(p, True), x, step, bounds)
                reverse_checked = True
            _log(dict(header,event=getattr(reverse, 'engine', 'reverse_mps'),analytic_energy=analytic_energy,
                      derivative_target='analytic expectation',objective_shots='unchanged'))
        else:
            value = central_gradient(lambda p: evaluate(p, True), x, step, bounds)
        _log(dict(header, event='gradient', index=count['gradient'],
                  parameters=np.asarray(x).tolist(),parameter_count=len(x),
                  norm=float(np.linalg.norm(value)), seconds=time.perf_counter()-t))
        return value

    def accepted(x):
        count['accepted_iteration'] += 1
        if callback is not None:
            callback(x)

    status = 'interrupted_or_solver_early_stop'
    try:
        if not len(x0):
            result = OptimizeResult(x=np.asarray(x0), fun=evaluate(x0), nit=0,
                                    nfev=1, njev=0, success=True,
                                    message='No variational parameters')
        else:
            result = scipy_minimize(
                evaluate, np.asarray(x0, dtype=float), method='L-BFGS-B',
                jac=derivative, bounds=bounds, callback=accepted,
                options=dict(maxiter=cap, maxfun=cap,
                             ftol=float(os.environ.get('ANALYTIC_MPS_LBFGS_FTOL','1e-9')),
                             gtol=float(os.environ.get('ANALYTIC_MPS_LBFGS_GTOL','1e-3')),
                             maxls=40, maxcor=10))
        status = str(result.message)
        print(f'[gradient] finished: {status}; counts={count}', flush=True)
        return result
    except ImmediateEnergyStop:
        status = f'Immediate energy stopping: consecutive |delta E| < {immediate:g} (no patience/window)'
        print(f'[gradient] {status}; counts={count}; best={best[0]:.10g}', flush=True)
        return OptimizeResult(x=best[1], fun=best[0], nfev=count['objective'],
                              njev=count['gradient'], nit=count['accepted_iteration'],
                              success=True, message=status)
    finally:
        _log(dict(header, event='finish', status=status, counts=count,
                  seconds=time.perf_counter()-started))
