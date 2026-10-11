"""Common parameter/timing journal and topology-scoped warm restart.

Source checkpoints restore outer strategy topology. Pending optimizer journals
restore the latest inner parameters; SciPy's internal Hessian/simplex is rebuilt.
"""
from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from scipy.optimize import OptimizeResult

active = ContextVar('optimization_journal', default=None)
_lock = threading.Lock()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False)+'\n')
    os.replace(tmp, path)


class Journal:
    def __init__(self, fun, x0, optimizer):
        self.root = Path(os.environ.get('ANALYTIC_MPS_TRACE_DIR', '.'))
        self.call = uuid.uuid4().hex
        self.optimizer = optimizer
        self.name = getattr(fun, '__qualname__', type(fun).__name__)
        owner = getattr(fun, '__self__', None)
        cells = inspect.getclosurevars(fun).nonlocals if inspect.isfunction(fun) else {}
        owner = owner or cells.get('self')
        manager = getattr(owner, 'checkpoint_manager', None)
        identity = dict(objective=self.name, optimizer=optimizer, count=len(x0),
                        namespace=str(getattr(manager, 'checkpoint_path', 'unmanaged')),
                        shots=os.environ.get('VQE_SHOTS', '-1'),
                        entanglement=os.environ.get('VQE_ENTANGLEMENT', 'linear'))
        # Hamiltonian and adaptive operator order distinguish equal-sized solves.
        if owner is not None:
            for name in ('hamiltonian', 'selected_ops', 'operations_a', 'operations_b', 'blocks', 'partition_a', 'partition_b',
                         '_checkpoint_stage_idx'):
                value = getattr(owner, name, None)
                if value is not None: identity[name] = hashlib.sha256(str(value).encode()).hexdigest()
        for name in ('indices', 'subset', 'block_idx', 'block_index', 'stage_idx'):
            if name in cells: identity[name] = str(cells[name])
        if hasattr(fun, 'reverse_problem'):
            identity['objective_hamiltonian'] = hashlib.sha256(str(fun.reverse_problem[1]).encode()).hexdigest()
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.path = self.root/'optimizer_states'/(key+'.json')
        self.identity = identity
        self.state = dict(identity=identity, complete=False, x=np.asarray(x0).tolist())
        self.resumed = False
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if not saved.get('complete') and saved.get('identity') == identity:
                self.state = saved
                self.resumed = True
        self.iteration = int(self.state.get('iteration', 0))
        self.tick = time.perf_counter()
        self.cache = {}

    def log(self, event, **data):
        row = dict(event=event, optimizer_call=self.call, optimizer=self.optimizer,
                   objective=self.name, solve_key=self.path.stem, **data)
        path = self.root/'optimization_trace.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, path.open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')

    def save(self, **data):
        self.state.update(data)
        atomic(self.path, self.state)

    def accepted(self, point):
        now = time.perf_counter()
        x = np.asarray(point, dtype=float)
        self.iteration += 1
        self.log('iteration', iteration=self.iteration, seconds=now-self.tick,
                 parameters=x.tolist(), parameter_count=x.size,
                 energy=self.cache.get(x.tobytes()), complete=True)
        self.tick = now
        self.save(iteration=self.iteration, accepted_parameters=x.tolist())


def minimize(backend, fun, x0, *args, **kwargs):
    journal = Journal(fun, x0, backend.__name__)
    token = active.set(journal)
    x0 = np.asarray(journal.state['x'], dtype=float)
    original_callback = kwargs.pop('callback', None)
    best = [float('inf'), x0.copy()]
    previous = None
    tol = float(os.environ.get('ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL', '1'))
    if not np.isfinite(tol) or tol <= 0: raise ValueError('Positive energy tolerance required')
    class EnergyStop(Exception): pass

    @functools.wraps(fun)
    def measured(x, *fargs):
        nonlocal previous
        start = time.perf_counter()
        try:
            value = float(fun(x, *fargs))
        except BaseException as exc:
            point=np.asarray(x,dtype=float)
            journal.log('energy_evaluation_failed',seconds=time.perf_counter()-start,
                        parameters=point.tolist(),parameter_count=point.size,exception=type(exc).__name__)
            raise
        probe = backend.is_gradient_probe()
        point = np.asarray(x, dtype=float)
        if not np.isfinite(value): raise ValueError('Nonfinite objective')
        journal.log('energy_evaluation', kind='probe' if probe else 'objective',
                    seconds=time.perf_counter()-start, energy=value,
                    parameters=point.tolist(), parameter_count=point.size)
        if not probe:
            journal.cache[point.tobytes()] = value
            journal.save(x=point.tolist(), energy=value)
            if value < best[0]: best[:] = [value, point.copy()]
            delta = None if previous is None else abs(previous-value)
            previous = value
            # SPSA checks at its iteration boundary, after saving RNG and gains.
            if backend.__name__ != 'spsa_optimizer' and delta is not None and delta < tol:
                journal.log('energy_stop', absolute_energy_change=delta, tolerance=tol,
                            patience=0, window=0)
                # Backend records this completed evaluation before stopping.
        return value
    if getattr(fun, '__self__', None) is not None: measured.__self__ = fun.__self__
    def callback(point):
        journal.accepted(point)
        # These source callbacks implement obsolete rolling-window stopping.
        # Their checkpoints are already committed by every source objective.
        if os.environ.get('ANALYTIC_MPS_CENTRAL_STOP', '1') != '1' and original_callback:
            original_callback(point)
    journal.log('start', parameters=x0.tolist(), parameter_count=x0.size,
                resumed=journal.resumed, identity=journal.identity)
    try:
        try:
            result = backend.minimize(measured, x0, *args, callback=callback, **kwargs)
        except EnergyStop:
            result = OptimizeResult(x=best[1], fun=best[0], success=True,
                                    message=f'Consecutive objective |delta E| < {tol}; no patience/window')
        journal.save(complete=True)
        journal.log('finish', status=str(result.message))
        return result
    except Exception as exc:
        if type(exc).__name__ in ('ControlledEarlyStop','_ControlledEvaluationLimit','EarlyStop'):
            journal.save(complete=True)
        raise
    finally:
        journal.log('iteration_tail', seconds=time.perf_counter()-journal.tick,
                    iteration=journal.iteration+1, complete=False,
                    parameters=journal.state['x'], parameter_count=len(journal.state['x']),
                    energy=journal.state.get('energy'))
        active.reset(token)
