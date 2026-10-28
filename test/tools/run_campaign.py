"""One protein per worker; jobs start one at a time while the host stays responsive (admission.py).

Same scheduling, eviction/resume, cleanup and worker settings as the VQE_Docking matched
launcher (test_statevector_mps/tools/run_matched.py); admission state is shared with it.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import functools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from admission import Admission
from worker_resources import MemoryGuard, cleanup_group, gpu_cpus, kill_orphans, worker_preexec
from job_state import reserve_case, is_complete
from optimization_trace import atomic
from record_storage import prepare_record_storage, best_effort

ROOT=Path(__file__).resolve().parents[1]
THREAD_LIMITS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')
STOP_GRACE_SECONDS=60
ADMISSION_NAME='graph'


def select_jobs(args):
    jobs=json.loads((ROOT/'campaign.json').read_text())['jobs']
    for field,value in (('protein_id',args.proteins),('optimizer',args.optimizers),('entanglement',args.entanglements),('shots',args.shots)):
        if value is None:continue
        selected=set(value.split(','))
        unknown=selected-{str(j[field]) for j in jobs}
        if unknown:raise ValueError(f'Unknown {field}: {sorted(unknown)}')
        jobs=[j for j in jobs if str(j[field]) in selected]
    return sorted(jobs,key=lambda j:(-j['residues'],j['protein_id'],j['optimizer'],j['entanglement'],j['shots']))


def worker_environment(gpu):
    # Pinned to the GPU's CPU socket and low priority (worker_preexec); thread count not limited.
    env={k:v for k,v in os.environ.items() if k not in THREAD_LIMITS}
    # VQE_SUITE_ROOT/ANALYTIC_MPS_CAMPAIGN_GPU mark this campaign's workers for kill_orphans.
    return dict(env,VQE_SUITE_ROOT=str(ROOT),ANALYTIC_MPS_CAMPAIGN_GPU=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')


def stop_workers(pending,grace=STOP_GRACE_SECONDS):
    """SIGTERM every worker group; SIGKILL any group still alive after grace."""
    running=[p for p,*_ in pending.values() if p.poll() is None]
    for p in running:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+grace
    for p in running:
        try:p.wait(timeout=max(0,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f'Worker {p.pid} ignored SIGTERM for {grace}s; sending SIGKILL',flush=True)
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:p.wait(timeout=grace)
            except subprocess.TimeoutExpired:print(f'WARNING: worker {p.pid} could not be reaped (uninterruptible state)',flush=True)


def run(jobs,gpus,output):
    output.mkdir(parents=True,exist_ok=True)
    gpu_sockets={g:gpu_cpus(g) for g in gpus}
    pending={};active=collections.Counter();queue=collections.deque(jobs)
    done=[];skipped=[];launched=[];evicted=[];stopped=False
    requeued=set()  # cases of evicted jobs; a leftover lock holder must not drop them
    admission=None;last_status=0.0
    def status():
        atomic(output/'campaign_status.json',dict(gpus=gpus,admission='host responsiveness',host=dict(admission.last) if admission else {},
            running=len(pending),running_by_gpu={str(g):active[g] for g in gpus},queued=len(queue),finished=done,skipped=skipped,
            launched=len(launched),evicted=evicted,stopped=stopped,updated=time.time()))
    def stop(sig,frame):raise SystemExit(128+sig)
    with ExitStack() as stack:
        lock=stack.enter_context((output/'.campaign.lock').open('a'))
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError(f'Another launcher holds {output/".campaign.lock"}') from exc
        # SIGHUP: a closed terminal/tmux pane must stop and clean the workers too.
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            previous=signal.signal(sig,stop);stack.callback(signal.signal,sig,previous)
        kill_orphans([ROOT])
        memory=MemoryGuard(gpus,0)
        # Confirming RAM/GPU release can be slow when nvidia-smi is; do it off the scheduling loop.
        cleaner=stack.enter_context(ThreadPoolExecutor(max_workers=16,thread_name_prefix='cleanup'))
        admission=Admission(ADMISSION_NAME)
        stack.callback(admission.close)
        admission.start_watch()
        try:
            status()
            while pending or queue:
                memory.monitor(pending)
                changed=False
                for pid,(process,case,gpu,job) in list(pending.items()):
                    code=process.poll()
                    if code is None:continue
                    members=memory.group_members(process)
                    if pid in admission.evicted:
                        # Requeued: its memory_usage.json is written when the resumed run ends.
                        memory.peaks.pop(pid,None);memory.members.pop(pid,None)
                        cleaner.submit(cleanup_group,process,members)
                    else:
                        cleaner.submit(lambda p=process,case=case,members=members:memory.finished(p,case,cleanup_group(p,members)))
                    admission.ended(pid)
                    del pending[pid];active[gpu]-=1;changed=True
                    # Stopped by the admission watcher: requeue; it resumes from its checkpoint.
                    if pid in admission.evicted:
                        admission.evicted.discard(pid)
                        evicted.append(dict(case=str(case),gpu=gpu,evicted_at=time.time()))
                        queue.appendleft(job);requeued.add(case)
                        print(f'Evicted GPU {gpu}: {case.relative_to(output)}; requeued to resume later',flush=True)
                        continue
                    if code==0 and not is_complete(case):code=1
                    done.append(dict(case=str(case),gpu=gpu,exit_code=code))
                    print(f'Finished GPU {gpu}: {case.relative_to(output)}; status={code}',flush=True)
                gpu=admission.choose_gpu(gpu_sockets) if queue else None
                while gpu is not None and queue:
                    job=queue.popleft();case=output/job['key'];case.mkdir(parents=True,exist_ok=True)
                    case_lock,reason=reserve_case(case);changed=True
                    if reason=='running' and case in requeued:
                        # Its evicted processes are still being cleaned up: retry later.
                        queue.append(job);break
                    requeued.discard(case)
                    if reason:
                        skipped.append(dict(case=str(case),reason=reason));print(f'Skipping {job["key"]}: {reason}',flush=True);continue
                    try:
                        path=case/'job.json'
                        if path.exists() and json.loads(path.read_text())!=job:raise RuntimeError(f'Changed job: {case}')
                        atomic(path,job)
                        prepare_record_storage(case)
                        with (case/'run.log').open('a') as log:
                            p=subprocess.Popen([sys.executable,str(ROOT/'tools/run_case.py'),str(case),'--gpu',str(gpu),'--lock-fd',str(case_lock.fileno())],
                                cwd=case,env=worker_environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                pass_fds=(case_lock.fileno(),),preexec_fn=functools.partial(worker_preexec,gpu_sockets[gpu]))
                    finally:case_lock.close()
                    admission.started(p.pid,gpu,case)
                    pending[p.pid]=(p,case,gpu,job);active[gpu]+=1
                    launched.append(dict(job,pid=p.pid,gpu=gpu,log=str(case/'run.log'),started=time.time()))
                    print(f'Started GPU {gpu} ({active[gpu]} on this GPU, {len(pending)} total): {job["key"]}',flush=True)
                    break
                if changed:atomic(output/'launched_jobs.json',launched)
                if changed or time.time()-last_status>=30:status();last_status=time.time()
                if pending or queue:time.sleep(1)
        except BaseException:stopped=True;raise
        finally:
            stop_workers(pending)
            for process,*_ in pending.values():  # confirmed in parallel; the cleaner waits on exit
                cleaner.submit(cleanup_group,process,memory.group_members(process))
            pending.clear();active.clear();best_effort('final campaign status',status)
    return int(any(r['exit_code'] for r in done))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--gpus',default=os.environ.get('GRAPH_VQE_GPUS'),help='comma-separated GPU IDs (default: GRAPH_VQE_GPUS, else every GPU nvidia-smi lists)')
    p.add_argument('--proteins');p.add_argument('--optimizers');p.add_argument('--shots');p.add_argument('--entanglements')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    p.add_argument('--max-iter',type=int,default=1000000)
    args=p.parse_args()
    if args.gpus is None:
        args.gpus=','.join(subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split())
    gpus=tuple(int(g) for g in args.gpus.split(','))
    if not gpus or len(set(gpus))!=len(gpus) or min(gpus)<0 or args.max_iter<1:raise ValueError('Invalid GPU/cap configuration')
    jobs=select_jobs(args)
    print(f'{len(jobs)} cases; shots=-1/1000; COBYLA/L-BFGS-B/SPSA; linear/reverse_linear; GPUs={gpus}; jobs start one at a time while the '
          'host stays responsive and the GPU\'s CPU socket is not full; all jobs stop and requeue at 90% RAM; each pinned to the CPU socket '
          'next to its GPU at low priority, no thread limit; no RAM/GPU memory limits.',flush=True)
    print('One local solve per partition, then one global solve. Threshold 1, no patience/window. Completed cases are skipped.',flush=True)
    if args.dry_run:return 0
    import torch,opt_einsum
    if not torch.version.cuda:raise RuntimeError('CUDA-enabled PyTorch required')
    available=set(int(s) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split())
    if set(gpus)-available:raise ValueError('Requested GPUs unavailable; use --gpus with this server\'s GPU IDs')
    os.environ['GRAPH_VQE_MAX_ITER']=str(args.max_iter)
    return run(jobs,gpus,args.output.resolve())


if __name__=='__main__':raise SystemExit(main())
