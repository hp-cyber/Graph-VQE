import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import work_trace as work
import work_instrumentation as instrument
from work_report import write_report
from job_state import is_complete


class WorkRecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        work.start(self.root, method='independent-vqe', shots=-1)

    def tearDown(self):
        work.reset()
        self.tmp.cleanup()

    def rows(self):
        return [json.loads(line) for line in (self.root/'work_events.jsonl').read_text().splitlines()]

    def test_partition_and_global_records_preserve_values_and_calls(self):
        class Solver:
            blocks = [[0],[1,2]]
            num_qubits = 3
            nfev = 0
            def _evaluate_energy(self, params, **kwargs):
                self.nfev += 1
                return float(sum(params))
            def _minimize_model(self, params, model, phase, maxiter, qpu_id=0):
                value = self._evaluate_energy(params)
                with work.scope(role='gradient_probe'):
                    self._evaluate_energy(params + 1)
                return dict(energy=value, params=params)
            def _run_vqe(self):
                raise AssertionError('not needed')
        instrument.install(Solver)
        solver = Solver()
        for phase, qubits in [('local',[0]),('local',[1,2]),('global',[0,1,2])]:
            model = dict(qubits=qubits, width=max(2,len(qubits)))
            result = solver._minimize_model(np.array([2.,3.]), model, phase, 5)
            self.assertEqual(result['energy'], 5.)
        solver._evaluate_energy(np.array([4.,5.]))  # Real warm-start evaluation.
        self.assertEqual(solver.nfev, 7)
        work.stop()
        report = write_report(self.root)
        self.assertFalse(report['issues'])
        self.assertFalse(report['unclosed_events'])
        self.assertEqual(report['counts']['energy_evaluation'], 7)
        points = [r for r in self.rows() if r['event'] == 'global_energy']
        self.assertEqual([r['energy'] for r in points], [5.,9.])
        self.assertEqual([r['context']['stage_id'] for r in points], ['global','warm_start'])
        local = [r for r in self.rows() if r['event']=='local_solve' and r['phase']=='end']
        self.assertEqual([r['context']['partition_id'] for r in local], [0,1])
        self.assertEqual(local[0]['context']['qubits'], 1)
        self.assertEqual(local[0]['context']['circuit_qubits'], 2)

    def test_async_backend_recorded_once_in_parent_partition(self):
        result = types.SimpleNamespace(expectation=lambda: -2.)
        future = types.SimpleNamespace(get=lambda: result)
        cudaq = types.SimpleNamespace(observe=lambda *a,**k:result,
            observe_async=lambda *a,**k:future,sample=lambda *a,**k:{'0':1},sample_async=lambda *a,**k:future)
        with patch.dict(sys.modules,cudaq=cudaq),patch.dict(os.environ,VQE_SHOTS='1000'):
            instrument.install_backend()
            with work.scope(partition_id=2,stage_id='local'):
                with work.Span('energy_evaluation') as event:
                    pending = cudaq.observe_async('kernel','H',[.1,.2],shots_count=1000)
                    self.assertEqual(work._parent.get(),event.id)
                    event.result['energy'] = pending.get().expectation()
                    pending.get()
        rows = [r for r in self.rows() if r['event']=='backend_call' and r['phase']=='end']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['context']['partition_id'],2)
        self.assertEqual(rows[0]['parameter_count'],2)
        work.stop()
        self.assertFalse(write_report(self.root)['unclosed_events'])

    def test_completed_marker_cannot_hide_incomplete_recording(self):
        (self.root/'job_status.json').write_text('{"status":"complete"}')
        marker=self.root/'Result/process_data/strategy/protein/COMPLETED.json'
        marker.parent.mkdir(parents=True);marker.write_text('{}')
        self.assertFalse(is_complete(self.root))
        work.stop();write_report(self.root)
        self.assertTrue(is_complete(self.root))

    def test_truncated_tail_preserves_previous_attempt(self):
        work.point('saved_checkpoint',parameters=[.4])
        work.reset()
        with (self.root/'work_events.jsonl').open('ab') as f:f.write(b'{"partial":')
        work.start(self.root,method='independent-vqe',shots=-1)
        work.stop()
        report=write_report(self.root)
        self.assertEqual(len(report['attempts']),2)
        self.assertFalse(report['issues'])
        self.assertTrue(any(r['event']=='saved_checkpoint' for r in self.rows()))
        self.assertEqual(len(list(self.root.glob('work_truncated_tail_*.bin'))),1)


if __name__ == '__main__':
    unittest.main()
