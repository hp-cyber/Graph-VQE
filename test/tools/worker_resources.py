"""Process isolation, memory accounting and cleanup for GPU campaign workers.

No per-job RAM/GPU budgets, admission gating or memory-based termination are
applied: every worker may use whatever its host and GPU can provide. Memory
is only sampled for the record. When a worker exits, its whole process group
is killed and the launcher confirms that its RAM and GPU contexts are released.
"""
import os
from pathlib import Path
import signal
import subprocess
import time
from record_storage import atomic_json

GIB = 1024**3
RELEASE_SECONDS = 60
GPU_SAMPLE_SECONDS = 60
WORKER_NICE = 10  # CPU priority of workers: 0 = normal, 19 = lowest


def process_groups():
    groups = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            # Fields after comm: state, ppid, pgrp, ... rss (pages).
            group = int(fields[2])
            groups.setdefault(group, []).append((int(path.parent.name), int(fields[21])*os.sysconf('SC_PAGE_SIZE')))
        except (OSError, ValueError, IndexError):
            continue
    return groups


def gpu_processes():
    """PID -> bytes of every process that currently holds a CUDA context."""
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, stderr=subprocess.PIPE)
    usage = {}
    for line in output.splitlines():
        try:
            pid, mb = line.split(',')
            usage[int(pid)] = usage.get(int(pid), 0)+int(mb)*1024**2
        except ValueError:
            continue
    return usage


def cleanup_group(process, members=()):
    """Parent has exited/reaped; remove survivors and confirm memory release.

    RAM: wait until no member of the worker's process group keeps resident
    memory. GPU: the driver frees a context when its process exits; wait until
    no former member is listed by nvidia-smi. Each wait lasts at most
    RELEASE_SECONDS, so one unkillable process cannot stall the launcher.
    """
    pids = {process.pid, *members, *(pid for pid, _ in process_groups().get(process.pid, []))}
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # An unkillable (uninterruptible) process must not stall every other slot.
    deadline = time.monotonic()+RELEASE_SECONDS
    while True:
        alive = [pid for pid, rss in process_groups().get(process.pid, []) if rss]
        if not alive:
            break
        if time.monotonic() >= deadline:
            print(f'[memory] WARNING: PIDs {alive} of worker {process.pid} survive SIGKILL; continuing', flush=True)
            return dict(ram_released=False, gpu_released=None, surviving_pids=alive)
        time.sleep(.05)
    deadline = time.monotonic()+RELEASE_SECONDS
    while True:
        try:
            remaining = sorted(pids & set(gpu_processes()))
        except (OSError, subprocess.SubprocessError) as exc:
            print(f'[memory] Could not confirm GPU release for PID {process.pid} ({type(exc).__name__})', flush=True)
            return dict(ram_released=True, gpu_released=None)
        if not remaining:
            return dict(ram_released=True, gpu_released=True)
        if time.monotonic() >= deadline:
            print(f'[memory] WARNING: exited worker PIDs {remaining} still hold GPU contexts', flush=True)
            return dict(ram_released=True, gpu_released=False, gpu_pids_remaining=remaining)
        time.sleep(.2)


def gpu_cpus(gpu):
    """Logical CPUs of the CPU socket the GPU is attached to (from its PCI device).

    GPU numbers follow PCI bus order (CUDA_DEVICE_ORDER=PCI_BUS_ID, as nvidia-smi).
    Read from /proc, not nvidia-smi, which can take many seconds under load.
    Where this is unknown (no NUMA information, e.g. in some containers), every CPU this
    process may use.
    """
    try:
        bus = sorted(os.listdir('/proc/driver/nvidia/gpus'))[gpu]
        text = Path(f'/sys/bus/pci/devices/{bus}/local_cpulist').read_text().strip()
        cpus = set()
        for part in text.split(','):
            first, _, last = part.partition('-')
            cpus.update(range(int(first), int(last or first)+1))
    except (OSError, IndexError, ValueError):
        cpus = set()
    return frozenset(cpus & os.sched_getaffinity(0)) or frozenset(os.sched_getaffinity(0))


def worker_preexec(cpus=None):
    """Run in each worker before exec: SIGKILL it if the launcher dies, even by SIGKILL.

    Pins it to `cpus` (the socket next to its GPU), inherited by all its threads
    and children.

    Also lowers its CPU and disk priority (inherited by all its threads and
    children) so SSH, logging and system daemons stay responsive under load;
    this never limits a worker while the machine has spare CPU or disk time.
    """
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    os.nice(WORKER_NICE)
    if cpus:
        os.sched_setaffinity(0, cpus)
    ioprio_set = {'x86_64': 251, 'aarch64': 30}.get(os.uname().machine)
    if ioprio_set:
        libc.syscall(ioprio_set, 1, 0, (2 << 13) | 7)  # self: best-effort class, lowest level (ionice -c2 -n7)
    if os.getppid() == 1:  # launcher already gone
        os._exit(1)


def kill_orphans(suite_roots):
    """Kill workers (and their children) left by an earlier launcher of these suites.

    Called while holding the campaign lock, so no live launcher owns them. A
    process is a worker if its environment names one of these suites.
    """
    roots = {str(r).encode() for r in suite_roots}
    victims = []
    for path in Path('/proc').glob('[0-9]*'):
        pid = int(path.name)
        if pid == os.getpid():
            continue
        try:
            env = dict(v.split(b'=', 1) for v in (path/'environ').read_bytes().split(b'\0') if b'=' in v)
        except (OSError, ValueError):
            continue
        if env.get(b'VQE_SUITE_ROOT') in roots and b'ANALYTIC_MPS_CAMPAIGN_GPU' in env:
            victims.append(pid)
    if victims:
        print(f'[memory] Killing {len(victims)} orphaned worker processes from an earlier launcher: {victims}', flush=True)
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic()+RELEASE_SECONDS
    while victims and time.monotonic() < deadline:
        victims = [pid for pid in victims if Path(f'/proc/{pid}').exists()
                   and Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z']
        time.sleep(.2)
    if victims:
        print(f'[memory] WARNING: orphaned worker PIDs {victims} survive SIGKILL', flush=True)
    return victims


class MemoryGuard:
    """Records sampled per-job peaks; never limits, delays or kills a worker."""

    def __init__(self, gpus, slots):
        self.peaks = {}
        self.members = {}
        self.last_error_notice = None
        self.gpu_usage = {}
        self.next_gpu_sample = 0.0
        print('[memory] No per-job RAM/GPU limits; usage is sampled for memory_usage.json only.', flush=True)

    def monitor(self, pending):
        if not pending:
            return
        groups = process_groups()
        # nvidia-smi can take many seconds with many GPU contexts: sample GPU memory once a minute.
        if time.monotonic() >= self.next_gpu_sample:
            self.next_gpu_sample = time.monotonic()+GPU_SAMPLE_SECONDS
            try:
                self.gpu_usage = gpu_processes()
            except (OSError, subprocess.SubprocessError) as exc:
                now = time.monotonic()
                if self.last_error_notice is None or now-self.last_error_notice >= 30:
                    print(f'[memory] GPU usage sampling unavailable ({type(exc).__name__}); workers are unaffected.', flush=True)
                    self.last_error_notice = now
                self.gpu_usage = {}
        usage = self.gpu_usage
        for pid in pending:
            members = groups.get(pid, [])
            self.members.setdefault(pid, set()).update(p for p, _ in members)
            ram = sum(rss for _, rss in members)
            vram = sum(usage.get(p, 0) for p, _ in members)
            old = self.peaks.get(pid, (0, 0))
            self.peaks[pid] = (max(old[0], ram), max(old[1], vram))

    def group_members(self, process):
        return self.members.get(process.pid, set())

    def finished(self, process, case, released=None):
        ram, vram = self.peaks.pop(process.pid, (0, 0))
        self.members.pop(process.pid, None)
        record = dict(peak_sampled_ram_bytes=ram, peak_sampled_gpu_bytes=vram,
                      process_group_cleaned=True, **(released or {}), finished=time.time())
        atomic_json(case/'memory_usage.json', record)


def gpu_memory():
    """GPU -> (total, free) bytes. Used by graph_resources' feasibility check, never to limit a worker."""
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True, stderr=subprocess.PIPE)
    return {int(i): (int(total)*1024**2, int(free)*1024**2)
            for i, total, free in (line.split(',') for line in output.splitlines())}
