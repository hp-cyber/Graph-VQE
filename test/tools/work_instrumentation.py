"""Observe Graph-VQE work without changing its partitions, circuits or optimizer."""
import functools
import hashlib
import numpy as np
import work_trace as trace


def kernel_metadata(kernel):
    return dict(circuit_id=hashlib.sha256(str(kernel).encode()).hexdigest())


def optimizer_context(fun):
    # The enclosing local/global solve supplies the exact model and partition.
    return {}


def install(cls):
    if getattr(cls, '_graph_work_instrumented', False):
        return
    original_solve = cls._minimize_model
    @functools.wraps(original_solve)
    def solve(self, params, model, phase, maxiter, qpu_id=0):
        partition = self.blocks.index(model['qubits']) if phase == 'local' else None
        fields = dict(stage_id=phase, partition_id=partition, logical_qubits=list(model['qubits']),
                      qubits=len(model['qubits']), circuit_qubits=model['width'], role='initialization',
                      energy_scope='local_hamiltonian' if phase == 'local' else 'full_hamiltonian')
        with trace.scope(**fields):
            with trace.Span('local_solve' if phase == 'local' else 'global_solve',
                            parameters=np.asarray(params).tolist(), parameter_count=len(params)) as event:
                before = getattr(self, 'nfev', 0)
                result = original_solve(self, params, model, phase, maxiter, qpu_id)
                event.result['reused_completed_solve'] = getattr(self, 'nfev', 0) == before
                event.result.update(retained_energy=float(result['energy']),
                                    retained_parameters=np.asarray(result['params']).tolist())
                trace.point('retained_state', energy=float(result['energy']),
                            parameters=np.asarray(result['params']).tolist(), parameter_count=len(result['params']))
                return result
    cls._minimize_model = solve

    original_energy = cls._evaluate_energy
    @functools.wraps(original_energy)
    def energy(self, params, *args, **kwargs):
        inherited = trace.context()
        fields = dict(stage_id=inherited.get('stage_id', 'warm_start'),
                      energy_scope=inherited.get('energy_scope', 'full_hamiltonian'),
                      role=inherited.get('role', 'initialization'),
                      qubits=inherited.get('qubits', self.num_qubits),
                      circuit_qubits=inherited.get('circuit_qubits', max(2, self.num_qubits)),
                      logical_qubits=inherited.get('logical_qubits', list(range(self.num_qubits))))
        with trace.scope(**fields):
            with trace.Span('energy_evaluation', parameters=np.asarray(params).tolist(), parameter_count=len(params)) as event:
                value = original_energy(self, params, *args, **kwargs)
                event.result['energy'] = float(value)
            if fields['energy_scope'] == 'full_hamiltonian' and fields['role'] not in ('gradient_probe', 'calibration_probe', 'gradient'):
                trace.point('global_energy', energy=float(value), energy_scope='full_hamiltonian',
                            parameters=np.asarray(params).tolist(), parameter_count=len(params),
                            source_event_id=event.id, construction='evaluated_global_state')
            return value
    cls._evaluate_energy = energy

    original_run = cls._run_vqe
    @functools.wraps(original_run)
    def run(self, *args, **kwargs):
        trace.point('partition_plan', partitions=[list(b) for b in self.blocks], independent=False,
                    crossing_term_count=len(self.cross_terms), local_parameter_counts=[m['count'] for m in self.local_models],
                    global_parameter_count=self.num_params, global_ansatz='nearest_neighbor_su2_v1')
        with trace.Span('strategy') as event:
            result = original_run(self, *args, **kwargs)
            event.result.update(retained_energy=float(result[3][0][0]),
                                retained_parameters=np.asarray(result[1]).tolist())
            return result
    cls._run_vqe = run
    cls._graph_work_instrumented = True


def install_backend():
    import cudaq
    for name in ('observe','observe_async','sample','sample_async'):
        original=getattr(cudaq,name)
        def make(original,name):
            @functools.wraps(original)
            def call(*args,**kwargs):
                if not trace.enabled():return original(*args,**kwargs)
                params=np.asarray(args[1:] if name.startswith('sample') else args[2:],dtype=float).reshape(-1).tolist()
                info=kernel_metadata(args[0])
                with trace.scope(**info):
                    span=trace.Span('backend_call',operation=name,parameters=params,parameter_count=len(params),
                        shots=int(__import__('os').environ.get('VQE_SHOTS',kwargs.get('shots_count',-1))),
                        requested_shots=kwargs.get('shots_count'))
                    span.__enter__()
                    try:result=original(*args,**kwargs)
                    except BaseException as exc:
                        span.__exit__(type(exc),exc,exc.__traceback__);raise
                    if name.endswith('_async'):
                        trace._parent.reset(span.token);span.token=None
                        class Future:
                            def __init__(self):self.done=False
                            def get(self):
                                try:out=result.get()
                                except BaseException as exc:
                                    if not self.done:span.__exit__(type(exc),exc,exc.__traceback__);self.done=True
                                    raise
                                if not self.done:
                                    if name.startswith('observe'):span.result['observable_expectation']=float(out.expectation())
                                    span.__exit__(None,None,None);self.done=True
                                return out
                            def __getattr__(self,key):return getattr(result,key)
                        return Future()
                    if name.startswith('observe'):span.result['observable_expectation']=float(result.expectation())
                    span.__exit__(None,None,None)
                    return result
            return call
        setattr(cudaq,name,make(original,name))


def gradient(fun):
    @functools.wraps(fun)
    def measured(x):
        with trace.scope(role='gradient'):
            with trace.Span('gradient_evaluation',parameters=np.asarray(x).tolist(),parameter_count=len(x)) as span:
                result=fun(x)
                span.result['gradient']=np.asarray(result).tolist()
                return result
    return measured
