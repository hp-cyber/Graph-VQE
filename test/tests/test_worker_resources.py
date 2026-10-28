import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import worker_resources as resources
from clean_unfinished import clean
from job_state import reserve_case


class ResourceTests(unittest.TestCase):
    def test_gpu_queries_have_no_time_limit(self):
        with patch.object(resources.subprocess,'check_output',return_value='2, 40000, 39000\n') as query:
            resources.gpu_memory()
            self.assertNotIn('timeout',query.call_args.kwargs)

    def test_monitor_errors_preserve_workers(self):
        failed=subprocess.CalledProcessError(1,['nvidia-smi'])
        with tempfile.TemporaryDirectory() as d,patch.object(resources,'process_groups',return_value={123:[(123,resources.GIB)]}),patch.object(resources.subprocess,'check_output',side_effect=failed),patch.object(resources.os,'killpg') as kill:
            guard=resources.MemoryGuard((2,),0)
            guard.monitor({123:(None,Path(d),2)})
            kill.assert_not_called()
            self.assertEqual(guard.peaks[123],(resources.GIB,0))

    def test_finished_parent_descendant_memory_is_released(self):
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'child'
            child="import time; memory=bytearray(32*1024**2); time.sleep(60)"
            code=f"import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c',{child!r}]); Path({str(marker)!r}).write_text(str(p.pid))"
            p=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                p.wait()
                while not any(rss for _,rss in resources.process_groups().get(p.pid,[])):time.sleep(.02)
                self.assertTrue(resources.process_groups().get(p.pid))
                resources.cleanup_group(p)
                self.assertFalse(any(rss for _,rss in resources.process_groups().get(p.pid,[])))
            finally:
                try:os.killpg(p.pid,9)
                except ProcessLookupError:pass

    def test_memory_is_recorded_never_limited(self):
        gib=resources.GIB
        with tempfile.TemporaryDirectory() as d,patch.object(resources,'process_groups',return_value={123:[(123,500*gib)]}),patch.object(resources.subprocess,'check_output',return_value='123, 400000\n'),patch.object(resources.os,'killpg') as kill:
            guard=resources.MemoryGuard((2,),0)
            guard.monitor({123:(None,Path(d),2)})
            kill.assert_not_called()
            self.assertFalse((Path(d)/'resource_failure.json').exists())
            guard.finished(unittest.mock.Mock(pid=123),Path(d),dict(ram_released=True,gpu_released=True))
            record=json.loads((Path(d)/'memory_usage.json').read_text())
            self.assertEqual((record['peak_sampled_ram_bytes'],record['peak_sampled_gpu_bytes']),(500*gib,400000*1024**2))

    def test_clean_preserves_completed_and_locked_cases(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('complete','failed','partial','running'):
                p=root/name;p.mkdir();(p/'job.json').write_text('{}');(p/'checkpoint.npy').write_bytes(b'parameters')
            (root/'complete/job_status.json').write_text('{"status":"complete"}')
            (root/'failed/job_status.json').write_text('{"status":"failed"}')
            lock,_=reserve_case(root/'running')
            try:rows=clean([root],True)
            finally:lock.close()
            self.assertEqual(sum(r['action']=='cleaned' for r in rows),2)
            for name in ('complete','running'):self.assertTrue((root/name/'checkpoint.npy').exists())
            for name in ('failed','partial'):
                self.assertFalse((root/name/'checkpoint.npy').exists())
                self.assertTrue((root/name/'job.json').exists())


if __name__=='__main__':unittest.main()
