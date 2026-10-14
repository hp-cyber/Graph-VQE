"""Keep growing records on optional data storage without moving checkpoints."""
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

RECORD_FILES = ('work_events.jsonl', 'optimization_trace.jsonl',
    'backend_evaluations.jsonl', 'energy_evaluations.csv',
    'optimizer_iterations.csv', 'global_energy_timeline.jsonl',
    'cobyla_evaluations.jsonl', 'lbfgs_evaluations.jsonl',
    'spsa_evaluations.jsonl', 'run.log')


def atomic_json(path, data):
    """A failed write leaves the previous committed JSON untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as f:
            f.write(payload)
        os.replace(temporary, path)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass
        except OSError: pass  # Do not replace the original disk error.


def best_effort(label, operation, *args):
    """Shutdown reporting must never replace the failure that caused shutdown."""
    try:
        return operation(*args)
    except OSError as exc:
        print(f'[storage] Could not save {label}: {exc}', file=sys.stderr, flush=True)


def configured_root():
    value = os.environ.get('CAMPAIGN_RECORDS_ROOT')
    return Path(value).expanduser().resolve() if value else None


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''): digest.update(block)
    return digest.hexdigest()


def repair_append_logs(case):
    """Archive an interrupted final JSONL write before another worker appends.

    The case lock must be held. Work events have their own recovery/audit in
    work_trace; generated CSV/timeline exports are rebuilt by work_report.
    """
    case = Path(case)
    recovered = []
    names = ('optimization_trace.jsonl', 'backend_evaluations.jsonl',
             'cobyla_evaluations.jsonl', 'lbfgs_evaluations.jsonl',
             'spsa_evaluations.jsonl')
    for name in names:
        path = case/name
        if not path.exists() or not path.stat().st_size: continue
        with path.open('rb+') as f:
            size = f.seek(0, 2); f.seek(size-1)
            if f.read(1) == b'\n': continue
            position = size; chunks = []
            while position:
                length = min(position, 65536); position -= length
                f.seek(position); chunk = f.read(length); chunks.append(chunk)
                if b'\n' in chunk: break
            tail = b''.join(reversed(chunks)).rsplit(b'\n', 1)[-1]
            try: json.loads(tail)
            except (ValueError, UnicodeDecodeError):
                archive = case/'record_recovery'/(name+'.'+uuid.uuid4().hex+'.bin')
                archive.parent.mkdir(parents=True, exist_ok=True)
                with archive.open('xb') as saved:
                    saved.write(tail); saved.flush(); os.fsync(saved.fileno())
                record = dict(file=name, bytes=len(tail), archive=str(archive.relative_to(case)),
                              action='archive_incomplete_final_record')
                atomic_json(archive.with_suffix('.json'), record)
                f.truncate(size-len(tail))
            else:
                f.seek(0, 2); f.write(b'\n')
                record = dict(file=name, action='finish_valid_final_record')
            recovered.append(record)
    return recovered


def prepare_record_storage(case, root=None):
    """Call only with the case lock held and no running worker for this case.

    Copy and verify each existing log before atomically replacing its directory
    entry with a link. The case directory and all checkpoint paths stay put.
    """
    case = Path(case).resolve()
    manifest = case/'record_storage.json'
    saved = json.loads(manifest.read_text()) if manifest.exists() else None
    root = Path(root).expanduser().resolve() if root is not None else configured_root()
    if saved:
        if saved['case'] != str(case): raise ValueError('Record storage belongs to a different case')
        directory = Path(saved['directory'])
    elif root is not None:
        directory = root/hashlib.sha256(str(case).encode()).hexdigest()
    else:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    owner = directory/'case_path.json'
    if owner.exists() and json.loads(owner.read_text()) != {'case': str(case)}:
        raise ValueError('External record directory belongs to a different case')
    atomic_json(owner, {'case': str(case)})
    record = dict(schema='vqe_record_storage.v1', case=str(case), directory=str(directory), files=list(RECORD_FILES))
    atomic_json(manifest, record)
    for name in RECORD_FILES:
        source = case/name; target = directory/name
        if source.is_symlink():
            if source.resolve() != target.resolve() or not target.is_file():
                raise ValueError(f'Broken or unexpected record link: {source}')
            continue
        if source.exists():
            if not source.is_file(): raise ValueError(f'Expected a record file: {source}')
            if target.exists():
                if checksum(source) != checksum(target):
                    raise ValueError(f'Conflicting external record: {target}; both files preserved')
            else:
                temporary = target.with_name(name + '.' + uuid.uuid4().hex + '.copy')
                try:
                    shutil.copyfile(source, temporary)
                    with temporary.open('rb') as f: os.fsync(f.fileno())
                    if checksum(source) != checksum(temporary):
                        raise IOError(f'Record copy checksum mismatch: {source}')
                    os.replace(temporary, target)
                finally:
                    try: temporary.unlink()
                    except FileNotFoundError: pass
        elif not target.exists():
            target.touch(exist_ok=False)
        link = source.with_name(name + '.' + uuid.uuid4().hex + '.link')
        try:
            link.symlink_to(target)
            os.replace(link, source)
        finally:
            try: link.unlink()
            except FileNotFoundError: pass
    return record


def remove_record_storage(case):
    """Remove only this case's explicitly owned external records during cleanup."""
    case = Path(case).resolve(); manifest = case/'record_storage.json'
    if not manifest.exists(): return
    record = json.loads(manifest.read_text());directory = Path(record['directory'])
    if record['case'] != str(case) or directory.name != hashlib.sha256(str(case).encode()).hexdigest():
        raise ValueError('Refusing to clean an unrelated record directory')
    owner = directory/'case_path.json'
    if json.loads(owner.read_text()) != {'case': str(case)}:
        raise ValueError('External record ownership does not match')
    for name in RECORD_FILES:
        target = directory/name; source = case/name
        if source.is_symlink() and source.resolve() == target.resolve():
            target.unlink(missing_ok=True)
    owner.unlink()
    try: directory.rmdir()
    except OSError as exc:
        if exc.errno != errno.ENOTEMPTY: raise
