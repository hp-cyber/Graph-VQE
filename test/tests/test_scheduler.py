import json
import errno
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_campaign
from optimization_trace import atomic


class FakeAdmission:
    """Admits one job per call on the GPU with the fewest jobs; evicts the PIDs in `evict`."""
    def __init__(self,gpus):
        self.gpus=gpus;self.last={};self.evicted=set();self.evict=set();self.jobs={}
    def choose_gpu(self,sockets):
        load={g:sum(j==g for j in self.jobs.values()) for g in sockets}
        return min(sockets,key=lambda g:(load[g],g))
    def started(self,pid,gpu,case):self.jobs[pid]=gpu
    def ended(self,pid):self.jobs.pop(pid,None)
    def jobs_on_gpu(self,gpu,exclude=()):return sum(g==gpu for p,g in self.jobs.items() if p not in exclude)
    def start_watch(self):pass
    def close(self):pass


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        guard=patch.object(run_campaign,'MemoryGuard').start()
        guard.return_value.group_members.return_value=set()
        patch.object(run_campaign,'cleanup_group').start()
        patch.object(run_campaign,'kill_orphans').start()
        patch.object(run_campaign,'gpu_cpus',side_effect=lambda g:frozenset({0})).start()
        patch.object(run_campaign.os,'sched_getaffinity',return_value={0}).start()
        self.admission=FakeAdmission((2,3))
        patch.object(run_campaign,'Admission',return_value=self.admission).start()
        self.addCleanup(patch.stopall)

    def test_failed_records_do_not_stop_the_launcher(self):
        # memory_usage.json is written by a cleanup thread; a full disk there or in the
        # final status must not stop other jobs or hide the worker's own exit status.
        run_campaign.MemoryGuard.return_value.finished.side_effect=OSError(errno.ENOSPC,'finished record is full')
        original=run_campaign.atomic
        def save(path,data):
            if isinstance(data,dict) and data.get('stopped'):raise OSError(errno.ENOSPC,'final status is full')
            return original(path,data)
        worker=unittest.mock.Mock(pid=543210);worker.poll.return_value=0
        with tempfile.TemporaryDirectory() as d, patch.object(run_campaign,'atomic',side_effect=save), patch.object(run_campaign.subprocess,'Popen',return_value=worker):
            code=run_campaign.run([dict(key='case0')],(2,),Path(d))
            self.assertEqual(code,1)  # no job_status.json: not complete
            self.assertTrue(run_campaign.cleanup_group.called)
            self.assertTrue(run_campaign.MemoryGuard.return_value.finished.called)

    def test_refill_skip_complete_continue_after_failure_and_requeue_evicted(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)
            jobs=[dict(key=f'case{i}',protein_id=str(i)) for i in range(12)]
            atomic(output/'case0/job_status.json',dict(status='complete'))
            instances=[];events=[];admission=self.admission
            class Process:
                def __init__(self,command,**kwargs):
                    self.case=Path(command[2]);self.gpu=int(command[4]);self.pid=len(instances)+10000
                    self.remaining=3;self.done=False;self.code=None
                    instances.append(self);events.append(('start',self.case.name,self.gpu))
                    if self.case.name=='case2' and not any(e==('evicted','case2') for e in events):
                        admission.evicted.add(self.pid);events.append(('evicted','case2'))  # watcher stops it
                def poll(self):
                    if not self.done:
                        self.remaining-=1
                        if self.remaining>0:return None
                        self.done=True;events.append(('finish',self.case.name,self.gpu))
                        if self.pid in admission.evicted:self.code=-15
                        else:
                            self.code=1 if self.case.name=='case1' else 0
                            atomic(self.case/'job_status.json',dict(status='failed' if self.code else 'complete'))
                    return self.code
            with patch.object(run_campaign.subprocess,'Popen',Process),patch.object(run_campaign.time,'sleep'), \
                 patch.object(run_campaign,'is_complete',side_effect=lambda c:json.loads((c/'job_status.json').read_text())['status']=='complete' if (c/'job_status.json').exists() else False):
                code=run_campaign.run(jobs,(2,3),output)
            self.assertEqual(code,1)
            self.assertEqual(len(instances),12)  # 11 jobs, case2 twice
            self.assertEqual(sum(p.case.name=='case2' for p in instances),2)
            self.assertEqual({p.gpu for p in instances},{2,3})
            status=json.loads((output/'campaign_status.json').read_text())
            self.assertEqual(status['running'],0)
            self.assertEqual(len(status['finished']),11)
            self.assertEqual(len(status['skipped']),1)
            self.assertEqual(len(status['evicted']),1)

    def run_oom(self,jobs,gpus):
        """case0 exits with a CUDA OOM message on its first run; every other run completes."""
        instances=[]
        class Process:
            def __init__(self,command,stdout=None,**kwargs):
                self.case=Path(command[2]);self.pid=len(instances)+20000;self.polls=0;self.code=None
                self.oom=self.case.name=='case0' and not any(p.case.name=='case0' for p in instances)
                self.log=stdout;instances.append(self)
                if self.oom:stdout.write('RuntimeError: CUDA error: out of memory\n');stdout.flush()
            def poll(self):
                self.polls+=1
                if self.polls<(2 if self.oom else 4):return None
                if self.code is None:
                    self.code=1 if self.oom else 0
                    atomic(self.case/'job_status.json',dict(status='failed' if self.oom else 'complete'))
                return self.code
        with tempfile.TemporaryDirectory() as d,patch.object(run_campaign.subprocess,'Popen',Process),patch.object(run_campaign.time,'sleep'), \
             patch.object(run_campaign,'is_complete',side_effect=lambda c:(c/'job_status.json').exists() and json.loads((c/'job_status.json').read_text())['status']=='complete'):
            code=run_campaign.run(jobs,gpus,Path(d))
            return code,instances,json.loads((Path(d)/'campaign_status.json').read_text())

    def test_gpu_oom_on_shared_gpu_is_requeued(self):
        code,instances,status=self.run_oom([dict(key='case0'),dict(key='case1')],(2,))
        self.assertEqual(code,0)
        self.assertEqual([p.case.name for p in instances],['case0','case1','case0'])
        self.assertEqual(len(status['oom_requeued']),1)
        self.assertEqual(sorted(r['exit_code'] for r in status['finished']),[0,0])

    def test_gpu_oom_alone_on_its_gpu_fails(self):
        code,instances,status=self.run_oom([dict(key='case0')],(2,))
        self.assertEqual(code,1)
        self.assertEqual(len(instances),1)
        self.assertEqual(status['oom_requeued'],[])

    def test_oom_markers_only_in_this_runs_output(self):
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/'run.log';log.write_text('torch.OutOfMemoryError: CUDA out of memory\n')
            self.assertTrue(run_campaign.gpu_oom(log,0))
            self.assertFalse(run_campaign.gpu_oom(log,log.stat().st_size))
            log.write_text('ValueError: bad input\n')
            self.assertFalse(run_campaign.gpu_oom(log,0))


class GpuMemoryAdmissionTests(unittest.TestCase):
    def test_no_start_on_gpu_with_less_than_ten_percent_free(self):
        import admission
        with tempfile.TemporaryDirectory() as d,patch.object(admission,'STATE_DIR',Path(d)):
            a=admission.Admission('test');a.last={'checked':1}
            with patch.object(a,'cpu_busy',return_value=.1),patch.object(admission,'gpu_free',return_value={0:.05,1:.5}):
                self.assertEqual(a.choose_gpu({0:{0},1:{1}}),1)
            with patch.object(admission,'START_INTERVAL',0),patch.object(a,'cpu_busy',return_value=.1), \
                 patch.object(admission,'gpu_free',return_value={0:.09,1:.099}):
                self.assertIsNone(a.choose_gpu({0:{0},1:{1}}))
            with patch.object(admission,'START_INTERVAL',0),patch.object(a,'cpu_busy',return_value=.1), \
                 patch.object(admission,'gpu_free',side_effect=subprocess.TimeoutExpired('nvidia-smi',30)):
                self.assertIsNone(a.choose_gpu({0:{0},1:{1}}))
            a.close()


if __name__=='__main__':unittest.main()
