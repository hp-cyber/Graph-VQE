"""Process isolation, memory admission and cleanup for GPU campaign workers.

Limits contain individual failures; they do not guarantee an allocation will
fit (other GPU users and allocations between polls can still cause OOM).
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from record_storage import atomic_json

GIB = 1024**3


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


def cleanup_group(process):
    """Parent has exited/reaped; remove any surviving same-session children."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    while any(rss for _, rss in process_groups().get(process.pid, [])):
        time.sleep(.05)


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True, stderr=subprocess.PIPE)
    return {int(i): (int(total)*1024**2, int(free)*1024**2)
            for i, total, free in (line.split(',') for line in output.splitlines())}


def host_memory():
    fields = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    total, available = fields['MemTotal'], fields['MemAvailable']
    # Respect cgroup v2 limits as well as host-wide MemAvailable.
    try:
        relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        group = Path('/sys/fs/cgroup')/relative.lstrip('/')
        while True:
            limit = (group/'memory.max').read_text().strip()
            if limit != 'max':
                limit = int(limit); current = int((group/'memory.current').read_text())
                total, available = min(total, limit), min(available, max(0, limit-current))
            if group == Path('/sys/fs/cgroup'): break
            group = group.parent
    except (OSError, ValueError, StopIteration):
        pass
    return total, available


class MemoryGuard:
    def __init__(self, gpus, slots):
        self.slots = slots
        self.selected_gpus = tuple(gpus)
        self.gpus = {}
        self.budgets = {}
        self.gpu_ok = False
        self.monitor_ok = True
        self.last_error_notice = None
        self.refresh()
        total, _ = host_memory()
        self.reserve = min(8*GIB, total*.1)
        self.ram_limit = int(float(os.environ.get('CAMPAIGN_JOB_RAM_GIB', total*.8/(len(gpus)*slots)/GIB))*GIB)
        if self.ram_limit <= 0:
            raise ValueError('Memory budgets must be positive')
        self.last_notice = 0
        self.peaks = {}
        print(f'[memory] Per-job GPU budgets: { {g: round(n/GIB,2) for g,n in self.budgets.items()} } GiB; RAM {self.ram_limit/GIB:.2f} GiB',flush=True)

    def refresh(self):
        try:
            current = gpu_memory()
            if any(g not in current for g in self.selected_gpus):
                raise ValueError('GPU memory query did not report every requested GPU')
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.gpu_ok = False
            self.query_failed(exc)
            return
        self.gpus = current
        for g in self.selected_gpus:
            self.budgets.setdefault(g, int(float(os.environ.get('CAMPAIGN_JOB_GPU_GIB', current[g][0]*.8/self.slots/GIB))*GIB))
        if min(self.budgets.values()) <= 0:
            raise ValueError('Memory budgets must be positive')
        self.gpu_ok = True

    def query_failed(self, exc):
        now=time.monotonic()
        if self.last_error_notice is None or now-self.last_error_notice >= 30:
            print(f'[memory] GPU monitoring unavailable ({type(exc).__name__}); running jobs continue, new launches wait for monitoring to recover.',flush=True)
            self.last_error_notice=now

    def environment(self, gpu):
        return dict(CAMPAIGN_JOB_GPU_BYTES=str(self.budgets[gpu]), CAMPAIGN_JOB_RAM_BYTES=str(self.ram_limit))

    def can_launch(self, gpu):
        if not self.gpu_ok or not self.monitor_ok:
            return False
        _, available = host_memory()
        total, free = self.gpus[gpu]
        allowed = free >= self.budgets[gpu]+total*.1 and available >= self.reserve+self.ram_limit
        if not allowed and time.monotonic()-self.last_notice > 30:
            print(f'[memory] Waiting for free memory on GPU {gpu}: {free/GIB:.2f} GiB GPU, {available/GIB:.2f} GiB RAM', flush=True)
            self.last_notice = time.monotonic()
        return allowed

    def monitor(self, pending):
        groups = process_groups()
        usage = {}
        self.monitor_ok = True
        if pending:
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True, stderr=subprocess.PIPE)
            except (OSError, subprocess.SubprocessError) as exc:
                self.monitor_ok = False
                self.query_failed(exc)
                output = ''
            for line in output.splitlines():
                try:
                    pid, mb = line.split(','); usage[int(pid)] = usage.get(int(pid), 0)+int(mb)*1024**2
                except ValueError:
                    continue
        for pid, (process, case, gpu) in pending.items():
            members = groups.get(pid, [])
            ram = sum(rss for _, rss in members)
            vram = sum(usage.get(p, 0) for p, _ in members)
            old = self.peaks.get(pid, (0,0))
            self.peaks[pid] = (max(old[0],ram),max(old[1],vram))
            if ram <= self.ram_limit and vram <= self.budgets[gpu]: continue
            record = dict(reason='worker memory budget exceeded',ram_bytes=ram,gpu_bytes=vram,
                          ram_budget_bytes=self.ram_limit,gpu_budget_bytes=self.budgets[gpu],time=time.time())
            (case/'resource_failure.json').write_text(json.dumps(record,indent=2)+'\n')
            print(f'[memory] Terminating {case}: {record}',flush=True)
            try: os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError: pass

    def finished(self, process, case):
        ram,vram=self.peaks.pop(process.pid,(0,0))
        record=dict(peak_sampled_ram_bytes=ram,peak_sampled_gpu_bytes=vram,
                    process_group_cleaned=True,finished=time.time())
        atomic_json(case/'memory_usage.json', record)
