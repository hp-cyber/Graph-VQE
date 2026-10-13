"""One protein per worker, common queue, immediate refill, five workers per GPU."""
import argparse
import collections
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from worker_resources import MemoryGuard, cleanup_group
from job_state import reserve_case, is_complete
from optimization_trace import atomic

ROOT=Path(__file__).resolve().parents[1]


def select_jobs(args):
    jobs=json.loads((ROOT/'campaign.json').read_text())['jobs']
    for field,value in (('protein_id',args.proteins),('optimizer',args.optimizers),('entanglement',args.entanglements),('shots',args.shots)):
        if value is None:continue
        selected=set(value.split(','))
        unknown=selected-{str(j[field]) for j in jobs}
        if unknown:raise ValueError(f'Unknown {field}: {sorted(unknown)}')
        jobs=[j for j in jobs if str(j[field]) in selected]
    return sorted(jobs,key=lambda j:(-j['residues'],j['protein_id'],j['optimizer'],j['entanglement'],j['shots']))


def stop_workers(pending):
    running=[p for p,_,_ in pending.values() if p.poll() is None]
    for p in running:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for p in running:
        p.wait()


def run(jobs,gpus,per_gpu,output):
    output.mkdir(parents=True,exist_ok=True)
    pending={};active=collections.Counter();queue=collections.deque(jobs)
    done=[];skipped=[];launched=[];stopped=False
    def status():
        atomic(output/'campaign_status.json',dict(running=len(pending),running_by_gpu={str(g):active[g] for g in gpus},
            queued=len(queue),finished=done,skipped=skipped,launched=len(launched),max_active_per_gpu=per_gpu,stopped=stopped,updated=time.time()))
    def stop(sig,frame):raise SystemExit(128+sig)
    with ExitStack() as stack:
        lock=stack.enter_context((output/'.campaign.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sig in (signal.SIGINT,signal.SIGTERM):
            previous=signal.signal(sig,stop);stack.callback(signal.signal,sig,previous)
        memory=MemoryGuard(gpus,per_gpu)
        try:
            status()
            while pending or queue:
                memory.refresh();memory.monitor(pending)
                changed=False
                for pid,(process,case,gpu) in list(pending.items()):
                    code=process.poll()
                    if code is None:continue
                    cleanup_group(process)
                    memory.finished(process,case)
                    if code==0 and not is_complete(case):code=1
                    done.append(dict(case=str(case),gpu=gpu,exit_code=code))
                    del pending[pid];active[gpu]-=1;changed=True
                    print(f'Finished GPU {gpu}: {case.relative_to(output)}; status={code}',flush=True)
                while queue and any(active[g]<per_gpu for g in gpus):
                    admitted=False
                    for gpu in gpus:
                        if active[gpu]>=per_gpu or not memory.can_launch(gpu):continue
                        while queue:
                            job=queue.popleft();case=output/job['key'];case.mkdir(parents=True,exist_ok=True)
                            lock,reason=reserve_case(case);changed=True
                            if reason:
                                skipped.append(dict(case=str(case),reason=reason));print(f'Skipping {job["key"]}: {reason}',flush=True);continue
                            try:
                                path=case/'job.json'
                                if path.exists() and json.loads(path.read_text())!=job:raise RuntimeError(f'Changed job: {case}')
                                atomic(path,job)
                                with (case/'run.log').open('a') as log:
                                    p=subprocess.Popen([sys.executable,str(ROOT/'tools/run_case.py'),str(case),'--gpu',str(gpu),'--lock-fd',str(lock.fileno())],
                                        cwd=case,env=dict(os.environ,**memory.environment(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                            finally:lock.close()
                            admitted=True
                            pending[p.pid]=(p,case,gpu);active[gpu]+=1
                            launched.append(dict(job,pid=p.pid,gpu=gpu,log=str(case/'run.log'),started=time.time()))
                            print(f'Started GPU {gpu} ({active[gpu]}/{per_gpu}): {job["key"]}',flush=True)
                            break
                    if not admitted:break
                if changed:atomic(output/'launched_jobs.json',launched);status()
                if pending or queue:time.sleep(1)
        except BaseException:stopped=True;raise
        finally:
            stop_workers(pending)
            for process,_,_ in pending.values():cleanup_group(process)
            pending.clear();active.clear();status()
    return int(any(r['exit_code'] for r in done))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--gpus',default=os.environ.get('GRAPH_VQE_GPUS','2,3,4,5,6'))
    p.add_argument('--jobs-per-gpu',type=int,default=5)
    p.add_argument('--proteins');p.add_argument('--optimizers');p.add_argument('--shots');p.add_argument('--entanglements')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    p.add_argument('--max-iter',type=int,default=1000000)
    args=p.parse_args();gpus=tuple(int(g) for g in args.gpus.split(','))
    if not gpus or len(set(gpus))!=len(gpus) or min(gpus)<0 or args.jobs_per_gpu<1 or args.max_iter<1:raise ValueError('Invalid GPU/cap configuration')
    jobs=select_jobs(args)
    print(f'{len(jobs)} cases; shots=-1/1000; COBYLA/L-BFGS-B/SPSA; linear/reverse_linear; GPUs={gpus}; {args.jobs_per_gpu}/GPU',flush=True)
    print('One local solve per partition, then one global solve. Threshold 1, no patience/window. Completed cases are skipped.',flush=True)
    if args.dry_run:return 0
    import torch,opt_einsum
    if not torch.version.cuda:raise RuntimeError('CUDA-enabled PyTorch required')
    available=set(int(s) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split())
    if set(gpus)-available:raise ValueError('Requested GPUs unavailable; use --gpus with this server\'s GPU IDs')
    os.environ['GRAPH_VQE_MAX_ITER']=str(args.max_iter)
    return run(jobs,gpus,args.jobs_per_gpu,args.output.resolve())


if __name__=='__main__':raise SystemExit(main())
