import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_campaign
from optimization_trace import atomic


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        guard=patch.object(run_campaign, 'MemoryGuard').start()
        guard.return_value.can_launch.return_value=True
        guard.return_value.environment.return_value={}
        patch.object(run_campaign, 'cleanup_group').start()
        self.addCleanup(patch.stopall)

    def test_refill_limits_skip_complete_and_continue_after_failure(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)
            jobs=[dict(key=f'case{i}',protein_id=str(i)) for i in range(10)]
            atomic(output/'case0/job_status.json',dict(status='complete'))
            instances=[];peak={0:0,1:0};events=[]
            class Process:
                def __init__(self,command,**kwargs):
                    self.case=Path(command[2]);self.gpu=int(command[4]);self.pid=len(instances)+10000
                    self.remaining=2 if self.gpu==0 else 5;self.done=False
                    instances.append(self);events.append(('start',self.case.name,self.gpu))
                    active=sum(not p.done and p.gpu==self.gpu for p in instances)
                    peak[self.gpu]=max(peak[self.gpu],active)
                    if active>2:raise AssertionError('Scheduler exceeded per-GPU limit')
                def poll(self):
                    if not self.done:
                        self.remaining-=1
                        if self.remaining>0:return None
                        self.done=True;events.append(('finish',self.case.name,self.gpu))
                        atomic(self.case/'job_status.json',dict(status='failed' if self.case.name=='case1' else 'complete'))
                    return 1 if self.case.name=='case1' else 0
            with patch.object(run_campaign.subprocess,'Popen',Process),patch.object(run_campaign.time,'sleep'):
                code=run_campaign.run(jobs,(0,1),2,output)
            self.assertEqual(code,1)
            self.assertEqual(len(instances),9)
            self.assertEqual(peak,{0:2,1:2})
            first_finish=next(i for i,event in enumerate(events) if event[0]=='finish')
            self.assertTrue(any(e[0]=='start' for e in events[first_finish+1:]))
            status=json.loads((output/'campaign_status.json').read_text())
            self.assertEqual(status['running'],0)
            self.assertEqual(len(status['finished']),9)
            self.assertEqual(len(status['skipped']),1)


if __name__=='__main__':unittest.main()
