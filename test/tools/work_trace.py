"""Append-only VQE work ledger. Times are wall times, never inferred from counts.

Start/end events survive interrupted attempts. Nested work carries parent IDs;
counts and timing levels must not be added together. No CUDA imports here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

_context = ContextVar('vqe_work_context', default={})
_parent = ContextVar('vqe_work_parent', default=None)
_lock = threading.RLock()
_ledger = None


def plain(value):
    if hasattr(value, 'tolist'): return plain(value.tolist())
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    if value is None or isinstance(value, (str,int,float,bool)): return value
    return str(value)


def context(): return dict(_context.get())


@contextmanager
def scope(**fields):
    token = _context.set(dict(_context.get(), **{k:plain(v) for k,v in fields.items() if v is not None}))
    try: yield
    finally: _context.reset(token)


class Ledger:
    def __init__(self, root, metadata):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'work_events.jsonl'
        self.recovered_tail = None
        if self.path.exists() and self.path.stat().st_size:
            # SIGKILL or power loss may interrupt the last write. Preserve that
            # fragment separately; never discard an earlier complete record.
            with self.path.open('rb+') as f:
                size=f.seek(0,2);f.seek(size-1)
                if f.read(1)!=b'\n':
                    pos=size;tail=b''
                    while pos and b'\n' not in tail:
                        chunk=min(pos,65536);pos-=chunk;f.seek(pos);tail=f.read(chunk)+tail
                    start=pos+tail.rfind(b'\n')+1
                    fragment=tail[tail.rfind(b'\n')+1:]
                    try:json.loads(fragment)
                    except (ValueError,UnicodeDecodeError):
                        recovered=self.root/('work_truncated_tail_'+uuid.uuid4().hex+'.bin')
                        recovered.write_bytes(fragment);f.truncate(start)
                        self.recovered_tail=dict(path=recovered.name,bytes=len(fragment))
                    else:f.seek(0,2);f.write(b'\n')
        self.attempt = uuid.uuid4().hex
        self.origin = time.perf_counter_ns()
        self.metadata = plain(metadata)
        self.sequence = 0
        self.blocks = {}
        self.block_count = 0
        self.block_epoch = None
        self.topologies = set()

    def append(self, row):
        with _lock:
            self.sequence += 1
            base = dict(schema='vqe_work.v1', attempt_id=self.attempt,
                       sequence=self.sequence, pid=os.getpid(), thread_id=threading.get_ident(),
                       method=self.metadata.get('method'), optimizer=self.metadata.get('optimizer'),
                       shots=self.metadata.get('shots'), protein_id=self.metadata.get('protein_id'),
                       gpu=self.metadata.get('gpu'))
            row = dict(base, **row)
            with self.path.open('a') as f:
                f.write(json.dumps(plain(row), allow_nan=False)+'\n')


def start(root, **metadata):
    global _ledger
    _ledger = Ledger(root, metadata)
    point('attempt_start', metadata=metadata)
    if _ledger.recovered_tail:point('truncated_record_recovered',**_ledger.recovered_tail)
    return _ledger


def stop(status='complete'):
    point('attempt_finish', status=status)


def enabled(): return _ledger is not None


def topology(operations):
    operations=plain(operations)
    key=hashlib.sha256(json.dumps(operations,sort_keys=True).encode()).hexdigest()
    if _ledger is not None:
        with _lock:
            if key not in _ledger.topologies:
                _ledger.topologies.add(key)
                point('ansatz_topology',topology_id=key,operations=operations)
    return key


def point(event, **fields):
    if _ledger is None: return
    with _lock:
        now = time.perf_counter_ns()
        _ledger.append(dict(event=event, phase='point', event_id=uuid.uuid4().hex,
                            parent_id=_parent.get(), monotonic_ns=now, unix_ns=time.time_ns(),
                            elapsed_seconds=(now-_ledger.origin)/1e9,
                            context=context(), **fields))


class Span:
    def __init__(self, event, **fields):
        self.event=event; self.fields=fields; self.result={}; self.id=uuid.uuid4().hex
        self.ledger=_ledger; self.token=None
    def __enter__(self):
        self.begin=time.perf_counter_ns(); self.wall=time.time_ns()
        self.parent=_parent.get(); self.ctx=context()
        if self.ledger is not None:
            self.ledger.append(dict(event=self.event, phase='start', event_id=self.id,
                parent_id=self.parent, start_monotonic_ns=self.begin, start_unix_ns=self.wall,
                start_elapsed_seconds=(self.begin-self.ledger.origin)/1e9,
                context=self.ctx, **self.fields))
        self.token=_parent.set(self.id)
        return self
    def __exit__(self, typ, exc, tb):
        end=time.perf_counter_ns()
        if self.token is not None: _parent.reset(self.token)
        if self.ledger is not None:
            self.ledger.append(dict(event=self.event, phase='end', event_id=self.id,
                parent_id=self.parent, start_monotonic_ns=self.begin, end_monotonic_ns=end,
                start_unix_ns=self.wall, end_unix_ns=time.time_ns(),
                start_elapsed_seconds=(self.begin-self.ledger.origin)/1e9,
                end_elapsed_seconds=(end-self.ledger.origin)/1e9,
                seconds=(end-self.begin)/1e9, status=('complete' if typ is None else
                    'early_stop' if typ.__name__ in ('ControlledEarlyStop','_ControlledEvaluationLimit','EarlyStop','EnergyStop') else
                    'interrupted' if issubclass(typ,(KeyboardInterrupt,SystemExit)) else 'failed'),
                exception=None if typ is None else typ.__name__, context=self.ctx,
                **self.fields, **self.result))
        return False


def interval(event, begin_ns, end_ns, **fields):
    if _ledger is None: return
    _ledger.append(dict(event=event, phase='end', event_id=uuid.uuid4().hex,
        parent_id=_parent.get(), start_monotonic_ns=begin_ns, end_monotonic_ns=end_ns,
        start_elapsed_seconds=(begin_ns-_ledger.origin)/1e9,
        end_elapsed_seconds=(end_ns-_ledger.origin)/1e9,
        seconds=(end_ns-begin_ns)/1e9, status='complete', context=context(), **fields))


def register_blocks(blocks):
    if _ledger is None: return
    with _lock:
        _ledger.block_count=len(blocks); _ledger.blocks={}; _ledger.block_epoch=None
    point('partition_plan', partitions=[list(b) for b in blocks], independent=True)


def block_energy(index, energy, *, epoch=None, parameters=None, initial=False):
    """Chronological best product-state energy; emit only once all parts are known.

    Initial values are observations, never fabricated or aligned by local index.
    An epoch starts from the prior retained best blocks, as SIEVE does.
    """
    if _ledger is None or not _ledger.block_count: return
    with _lock:
        if epoch is not None and epoch != _ledger.block_epoch:
            _ledger.block_epoch=epoch
        old=_ledger.blocks.get(int(index))
        if initial or old is None or energy < old['energy']:
            _ledger.blocks[int(index)]={'energy':float(energy),'parameters':plain(parameters),
                                       'topology_id':context().get('topology_id')}
        if len(_ledger.blocks)==_ledger.block_count:
            point('global_energy', energy=sum(v['energy'] for v in _ledger.blocks.values()),
                  energy_scope='full_hamiltonian', construction='sum_independent_block_best_measurements',
                  blocks={str(k):v['energy'] for k,v in _ledger.blocks.items()},
                  block_states={str(k):v for k,v in _ledger.blocks.items()},
                  all_partitions_available=True)


def reset():
    global _ledger
    _ledger=None
