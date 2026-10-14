"""Per-case completion checks and locks, also shared with direct workers."""
import fcntl
import json
import os


def is_complete(case):
    status = case/'job_status.json'
    if (case/'work_events.jsonl').exists():
        # Source markers precede postprocessing. A case is complete only once
        # its required accounting has also been finalized and checked.
        try:
            report=json.loads((case/'work_summary.json').read_text())
            return (json.loads(status.read_text()).get('status')=='complete'
                    and not report['issues'] and not report['unclosed_current_attempt'])
        except (FileNotFoundError,ValueError,KeyError):return False
    try:
        if json.loads(status.read_text()).get('status') == 'complete':
            return True
    except (FileNotFoundError, ValueError):
        pass
    return any((case/'Result/process_data').glob('*/*/COMPLETED.json'))


def reserve_case(case, inherited_fd=None):
    # A held flock, not a stale PID/status file, identifies a running worker.
    if inherited_fd is None:
        lock = (case/'.run.lock').open('a')
    else:
        lock = os.fdopen(inherited_fd, 'a')
        expected = (case/'.run.lock').stat()
        actual = os.fstat(lock.fileno())
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            lock.close()
            raise RuntimeError('Inherited lock does not belong to this case')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None, 'running'
    if is_complete(case):
        lock.close()
        return None, 'complete'
    return lock, None
