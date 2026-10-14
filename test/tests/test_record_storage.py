import errno
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import record_storage as storage
from clean_unfinished import clean


class RecordStorageTests(unittest.TestCase):
    def test_verified_move_preserves_checkpoint_paths_and_append(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)/'case';case.mkdir();external=Path(d)/'data'
            checkpoint=case/'checkpoint.json';checkpoint.write_text('{"parameters":[0.4]}')
            log=case/'work_events.jsonl';log.write_text('{"event":"old"}\n')
            before=storage.checksum(log)
            record=storage.prepare_record_storage(case, external)
            self.assertTrue(log.is_symlink());self.assertEqual(storage.checksum(log),before)
            self.assertFalse(checkpoint.is_symlink());self.assertEqual(checkpoint.resolve().parent,case)
            with log.open('a') as f:f.write('{"event":"new"}\n')
            self.assertEqual(len((Path(record['directory'])/'work_events.jsonl').read_text().splitlines()),2)
            self.assertEqual(storage.prepare_record_storage(case),record)

    def test_failed_copy_preserves_original_and_can_retry(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)/'case';case.mkdir();log=case/'work_events.jsonl';log.write_text('saved\n')
            with patch.object(storage.shutil,'copyfile',side_effect=OSError(errno.ENOSPC,'full')):
                with self.assertRaises(OSError):storage.prepare_record_storage(case,Path(d)/'data')
            self.assertFalse(log.is_symlink());self.assertEqual(log.read_text(),'saved\n')
            storage.prepare_record_storage(case)
            self.assertTrue(log.is_symlink());self.assertEqual(log.read_text(),'saved\n')

    def test_truncated_external_ledger_recovers_without_changing_checkpoint(self):
        import work_trace as work
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)/'case';case.mkdir()
            storage.prepare_record_storage(case,Path(d)/'data')
            checkpoint=case/'checkpoint.json';checkpoint.write_text('{"parameters":[0.2]}')
            try:
                work.start(case,method='test');work.point('saved_parameters',parameters=[0.2]);work.reset()
                with (case/'work_events.jsonl').open('ab') as f:f.write(b'{"interrupted":')
                work.start(case,method='test');work.stop()
                rows=[json.loads(line) for line in (case/'work_events.jsonl').read_text().splitlines()]
                self.assertTrue(any(r['event']=='saved_parameters' for r in rows))
                self.assertTrue(any(r['event']=='truncated_record_recovered' for r in rows))
                self.assertTrue((case/'work_events.jsonl').is_symlink())
                self.assertEqual(checkpoint.read_text(),'{"parameters":[0.2]}')
            finally:work.reset()

    def test_conflicting_external_record_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)/'case';case.mkdir()
            record=storage.prepare_record_storage(case,Path(d)/'data')
            log=case/'work_events.jsonl';log.unlink();log.write_text('different local record')
            with self.assertRaisesRegex(ValueError,'Conflicting'):storage.prepare_record_storage(case)
            self.assertEqual(log.read_text(),'different local record')
            self.assertEqual((Path(record['directory'])/'work_events.jsonl').read_text(),'')

    def test_interrupted_append_is_archived_and_valid_records_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)/'case';case.mkdir()
            storage.prepare_record_storage(case,Path(d)/'data')
            log=case/'optimization_trace.jsonl'
            log.write_bytes(b'{"energy":1}\n{"energy":')
            report=storage.repair_append_logs(case)
            self.assertEqual(len(report),1)
            self.assertEqual((case/report[0]['archive']).read_bytes(),b'{"energy":')
            self.assertTrue(log.is_symlink())
            with log.open('a') as f:f.write('{"energy":2}\n')
            self.assertEqual([json.loads(s) for s in log.read_text().splitlines()],
                             [{'energy':1},{'energy':2}])
            self.assertEqual(storage.repair_append_logs(case),[])

    def test_valid_last_record_without_newline_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d);log=case/'backend_evaluations.jsonl'
            log.write_bytes(b'{"seconds":1}')
            storage.repair_append_logs(case)
            self.assertEqual(log.read_bytes(),b'{"seconds":1}\n')

    def test_recovery_archive_failure_leaves_original_log_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d);log=case/'spsa_evaluations.jsonl'
            payload=b'{"energy":1}\n{"unfinished":';log.write_bytes(payload)
            with patch.object(storage,'atomic_json',side_effect=OSError(errno.ENOSPC,'full')):
                with self.assertRaises(OSError):storage.repair_append_logs(case)
            self.assertEqual(log.read_bytes(),payload)

    def test_atomic_write_failure_preserves_committed_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'status.json';storage.atomic_json(p,{'state':'committed'})
            with patch.object(storage.os,'replace',side_effect=OSError(errno.ENOSPC,'full')):
                with self.assertRaises(OSError):storage.atomic_json(p,{'state':'new'})
            self.assertEqual(json.loads(p.read_text()),{'state':'committed'})
            self.assertEqual(list(Path(d).glob('*.tmp')),[])

    def test_shutdown_reporting_preserves_primary_exception(self):
        primary=OSError(errno.ENOSPC,'primary disk error')
        def failed():raise OSError(errno.ENOSPC,'secondary status error')
        with self.assertRaises(OSError) as caught:
            try:raise primary
            finally:storage.best_effort('status',failed)
        self.assertIs(caught.exception,primary)

    def test_clean_removes_only_owned_external_records(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)/'cases'/'case';case.mkdir(parents=True)
            (case/'job.json').write_text('{}')
            record=storage.prepare_record_storage(case,Path(d)/'data')
            unrelated=Path(d)/'data'/'unrelated';unrelated.write_text('keep')
            clean([case.parent],True)
            self.assertFalse(Path(record['directory']).exists())
            self.assertEqual(unrelated.read_text(),'keep')
            self.assertTrue((case/'job.json').exists())


if __name__=='__main__':unittest.main()
