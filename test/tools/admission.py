"""Host-wide job admission by responsiveness, shared by every launcher on this host.

No job's cost is estimated. Launchers keep starting jobs, one at a time across
all launchers, while the host stays responsive. When it stops being responsive,
the most recently started running job (of any launcher) is stopped by its own
launcher and requeued; it resumes from its checkpoint later. No new job starts
until a running job finishes; if the host is still unresponsive after
EVICT_COOLDOWN, the next most recent job is stopped too.

Responsive means, for CHECKS_TO_ACT consecutive checks:
- a normal-priority probe (fork/exec /bin/true, fsync a file on /data) is fast;
- memory and disk stall pressure (/proc/pressure, "full" avg10) are low;
- MemAvailable is above MIN_AVAILABLE of total RAM;
- STATE_DIR/force_unresponsive does not exist (a manual switch to make every launcher back off).

Two more rules:
- A job is not started on a GPU whose CPU socket is full (>= CPU_FULL busy over
  the last CPU_WINDOW seconds); it goes to a GPU whose socket has room.
- When RAM use reaches RAM_KILL of total, every running job of every launcher is
  stopped and requeued; they resume from their checkpoints as starts continue.
  STATE_DIR/force_ram_full triggers this by hand.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

# Launchers that share this directory share one start rate and one eviction policy.
STATE_DIR = Path(os.environ.get('ANALYTIC_MPS_ADMISSION_DIR', Path(__file__).resolve().parents[1]/'admission'))
START_INTERVAL = float(os.environ.get('ANALYTIC_MPS_START_INTERVAL', '30'))  # seconds between starts, all launchers
EVICT_COOLDOWN = 60          # seconds before another eviction while still unresponsive
CHECK_INTERVAL = 2           # seconds between responsiveness checks
CHECKS_TO_ACT = 3            # consecutive bad checks before evicting
MAX_PROBE_SECONDS = 2.0      # fork/exec + fsync of a normal-priority probe
MAX_MEMORY_FULL = 10.0       # % of time all tasks stalled on memory (avg10)
MAX_IO_FULL = 50.0           # % of time all tasks stalled on disk (avg10)
MIN_AVAILABLE = 0.05         # fraction of RAM
CPU_FULL = float(os.environ.get('ANALYTIC_MPS_CPU_FULL', '0.95'))   # socket busy fraction counted as full
CPU_WINDOW = 10              # seconds of CPU usage averaged
RAM_KILL = float(os.environ.get('ANALYTIC_MPS_RAM_KILL', '0.90'))   # used RAM fraction that stops every job
EVICT_GRACE_SECONDS = 10     # SIGTERM -> SIGKILL for stopped jobs (they resume from checkpoints)


def pressure(resource):
    """'full' avg10 stall percentage; 0 where the kernel does not expose PSI."""
    try:
        text = Path(f'/proc/pressure/{resource}').read_text()
    except OSError:
        return 0.0
    for line in text.splitlines():
        if line.startswith('full'):
            return float(line.split()[1].split('=')[1])
    return 0.0


def meminfo():
    """MemTotal/MemAvailable in bytes, reduced to this cgroup's memory limit when one applies
    (inside a container /proc/meminfo reports the whole host)."""
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0])*1024
    total, available = values['MemTotal'], values['MemAvailable']
    try:
        relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        group = Path('/sys/fs/cgroup')/relative.lstrip('/')
        while True:
            limit = (group/'memory.max').read_text().strip() if (group/'memory.max').exists() else 'max'
            if limit != 'max':
                stat = dict(line.split() for line in (group/'memory.stat').read_text().splitlines())
                # Reclaimable page cache does not count as used, as for MemAvailable.
                used = int((group/'memory.current').read_text())-int(stat.get('inactive_file', 0))
                total, available = min(total, int(limit)), min(available, max(0, int(limit)-used))
            if group == Path('/sys/fs/cgroup'):
                break
            group = group.parent
    except (OSError, ValueError, StopIteration):
        pass
    values['MemTotal'], values['MemAvailable'] = total, available
    return values


def cpu_times():
    """CPU number -> (busy, total) jiffies; busy includes niced time, excludes idle and iowait."""
    times = {}
    for line in Path('/proc/stat').read_text().splitlines():
        if line.startswith('cpu') and line[3].isdigit():
            name, *fields = line.split()
            fields = list(map(int, fields))
            times[int(name[3:])] = (sum(fields)-fields[3]-fields[4], sum(fields))
    return times


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Admission:
    """One per launcher; host-wide state lives in STATE_DIR/state.json under a lock."""

    def __init__(self, name):
        self.name = name
        self.owner = os.getpid()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.bad = 0
        self.next_check = 0.0
        self.last = {}
        self.cpu_samples = []
        self.evicted = set()  # this launcher's jobs stopped by the watcher (requeue them on exit)
        with self.state() as s:
            s['jobs'] = {pid: j for pid, j in s['jobs'].items() if alive(j['owner'])}

    @contextmanager
    def state(self):
        with (STATE_DIR/'state.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = STATE_DIR/'state.json'
            s = json.loads(path.read_text()) if path.exists() else {}
            s.setdefault('jobs', {}); s.setdefault('frozen', False)
            s.setdefault('last_start', 0.0); s.setdefault('last_evict', 0.0)
            yield s
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(s, indent=1))
            os.replace(tmp, path)

    def responsive(self):
        start = time.monotonic()
        subprocess.run(['/bin/true'], check=True)
        with (STATE_DIR/f'probe.{self.owner}').open('w') as f:
            f.write(str(time.time())); f.flush(); os.fsync(f.fileno())
        probe = time.monotonic()-start
        memory = meminfo()
        available = memory['MemAvailable']/memory['MemTotal']
        self.last = dict(probe_seconds=round(probe, 3), memory_full=pressure('memory'),
                         io_full=pressure('io'), available_ram=round(available, 3), checked=time.time())
        if (STATE_DIR/'force_unresponsive').exists():  # manual/test switch: back off now
            return False
        return (probe <= MAX_PROBE_SECONDS and self.last['memory_full'] <= MAX_MEMORY_FULL
                and self.last['io_full'] <= MAX_IO_FULL and available >= MIN_AVAILABLE)

    def cpu_busy(self, cpus):
        """Busy fraction of these CPUs over the last CPU_WINDOW seconds (None until measured)."""
        if len(self.cpu_samples) < 2 or self.cpu_samples[-1][0]-self.cpu_samples[0][0] < CPU_WINDOW/2:
            return None
        (_, old), (_, new) = self.cpu_samples[0], self.cpu_samples[-1]
        busy = sum(new[c][0]-old[c][0] for c in cpus if c in new)
        total = sum(new[c][1]-old[c][1] for c in cpus if c in new)
        return busy/total if total else None

    def check(self):
        """Every CHECK_INTERVAL: sample CPU and RAM, update the bad-check streak; maybe mark evictions."""
        now = time.monotonic()
        if now < self.next_check:
            return
        self.next_check = now+CHECK_INTERVAL
        self.cpu_samples.append((now, cpu_times()))
        while now-self.cpu_samples[0][0] > CPU_WINDOW:
            self.cpu_samples.pop(0)
        memory = meminfo()
        used = 1-memory['MemAvailable']/memory['MemTotal']
        if (STATE_DIR/'force_ram_full').exists():  # manual/test switch: stop every job now
            used = 1.0
        if used >= RAM_KILL:
            self.bad = max(self.bad, 1)  # no new starts while RAM is this full
            self.last = dict(self.last, ram_used=round(used, 3), checked=time.time())  # replace, never mutate: read by other threads
            with self.state() as s:
                marked = [pid for pid, j in s['jobs'].items() if not j.get('evict') and alive(int(pid))]
                for pid in marked:
                    s['jobs'][pid]['evict'] = True
                if marked:
                    s['last_evict'] = time.time()
                    print(f'[admission] RAM {used:.0%} used >= {RAM_KILL:.0%}: stopping all {len(marked)} running jobs; '
                          'they are requeued and resume from their checkpoints', flush=True)
            return
        if self.responsive():
            self.bad = 0
            return
        self.bad += 1
        print(f'[admission] host not responsive ({self.bad}/{CHECKS_TO_ACT}): {self.last}', flush=True)
        if self.bad < CHECKS_TO_ACT:
            return
        with self.state() as s:
            if time.time()-s['last_evict'] < EVICT_COOLDOWN:
                return
            running = [(j['started'], pid) for pid, j in s['jobs'].items()
                       if not j.get('evict') and alive(int(pid))]
            if not running:
                return
            _, pid = max(running)
            s['jobs'][pid]['evict'] = True
            s['frozen'] = True
            s['last_evict'] = time.time()
            print(f'[admission] evicting most recent job {pid} ({s["jobs"][pid]["case"]}); '
                  'no new jobs until a running job finishes', flush=True)

    def start_watch(self):
        """Check the host and stop marked jobs in a background thread, so slow
        work in the launcher loop (e.g. nvidia-smi) never delays them."""
        threading.Thread(target=self._watch, name='admission-watch', daemon=True).start()

    def _watch(self):
        deadlines = {}
        while True:
            try:
                self.check()
                marked = self.evictions()
                for pid in marked-set(deadlines):
                    self.evicted.add(pid)
                    deadlines[pid] = time.monotonic()+EVICT_GRACE_SECONDS
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                for pid, deadline in list(deadlines.items()):
                    if pid not in marked:  # exited and removed by the launcher
                        del deadlines[pid]
                    elif time.monotonic() > deadline:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            except Exception as exc:
                self.bad = max(self.bad, 1)  # no new starts until a check succeeds
                print(f'[admission] WARNING: check failed ({type(exc).__name__}: {exc})', flush=True)
            time.sleep(CHECK_INTERVAL)

    def evictions(self):
        """PIDs of this launcher's jobs that must be stopped and requeued."""
        with self.state() as s:
            return {int(pid) for pid, j in s['jobs'].items() if j['owner'] == self.owner and j.get('evict')}

    def choose_gpu(self, gpu_cpus):
        """None unless a job may start now; else, among GPUs whose CPU socket is not full,
        the GPU with the fewest jobs. gpu_cpus: GPU -> its socket's CPUs."""
        if self.bad or not self.last:
            return None
        busy = {g: self.cpu_busy(cpus) for g, cpus in gpu_cpus.items()}
        self.last = dict(self.last, cpu_busy_by_gpu={g: None if b is None else round(b, 3) for g, b in busy.items()})
        gpus = [g for g, b in busy.items() if b is not None and b < CPU_FULL]
        if not gpus:
            return None
        with self.state() as s:
            running = [j for pid, j in s['jobs'].items() if not j.get('evict') and alive(int(pid))]
            if s['frozen'] and not running and time.time()-s['last_evict'] >= EVICT_COOLDOWN:
                s['frozen'] = False  # nothing left that could finish
                print('[admission] no jobs left running; starting new jobs again', flush=True)
            if s['frozen'] or time.time()-max(s['last_start'], s['last_evict']) < START_INTERVAL:
                return None
            s['last_start'] = time.time()  # claim this start slot before any other launcher checks
            load = {g: 0 for g in gpus}
            for j in s['jobs'].values():
                if j['gpu'] in load:
                    load[j['gpu']] += 1
        return min(gpus, key=lambda g: (load[g], g))

    def started(self, pid, gpu, case):
        with self.state() as s:
            s['jobs'][str(pid)] = dict(owner=self.owner, launcher=self.name, gpu=gpu, case=str(case), started=time.time())
            s['last_start'] = time.time()

    def ended(self, pid):
        """A job exited. A normal finish (not an eviction) lets new jobs start again."""
        with self.state() as s:
            job = s['jobs'].pop(str(pid), None)
            if job and not job.get('evict') and s['frozen']:
                s['frozen'] = False
                print('[admission] a job finished; starting new jobs again', flush=True)

    def close(self):
        with self.state() as s:
            s['jobs'] = {pid: j for pid, j in s['jobs'].items() if j['owner'] != self.owner}
            if not s['jobs']:
                s['frozen'] = False
        (STATE_DIR/f'probe.{self.owner}').unlink(missing_ok=True)
