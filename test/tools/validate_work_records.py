"""Audit completed Graph-VQE records, including every local and global solve."""
import json
from pathlib import Path
from work_report import write_report


def validate(case):
    case = Path(case)
    report = write_report(case)
    assert not report['issues'], report['issues']
    assert not report['unclosed_current_attempt'], report['unclosed_current_attempt']
    assert all(v['matched'] for v in report['reconciliation'].values()), report['reconciliation']
    assert json.loads((case/'job_status.json').read_text())['status'] == 'complete'
    partitions = json.loads((case/'partition.json').read_text())['blocks']
    seen = set()
    count = 0
    with (case/'work_events.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['phase'] != 'end' or row['status'] != 'complete':
                continue
            event = row['event'];ctx = row['context']
            if event in ('energy_evaluation','objective_call','optimizer_iteration','gradient_evaluation','backend_call'):
                assert row['parameter_count'] == len(row['parameters']), row
                assert row['seconds'] >= 0, row
            if event in ('energy_evaluation','objective_call','optimizer_iteration'):
                assert row['energy'] is not None, row
            if event == 'energy_evaluation' and ctx['stage_id'] == 'local':
                index = ctx['partition_id']
                assert ctx['logical_qubits'] == partitions[index], row
                assert ctx['qubits'] == len(partitions[index]), row
                assert ctx['circuit_qubits'] == max(2, len(partitions[index])), row
                seen.add(index)
            if event == 'energy_evaluation':
                count += 1
    # Earlier attempts can supply completed local solves on restart.
    assert seen == set(range(len(partitions))), (seen, partitions)
    points = [json.loads(line) for line in (case/'global_energy_timeline.jsonl').read_text().splitlines()]
    assert points
    assert all(p['energy_scope'] == 'full_hamiltonian' for p in points)
    assert all(p['context']['stage_id'] in ('warm_start','global') for p in points)
    times = [p['elapsed_active_seconds'] for p in points]
    assert times == sorted(times)
    assert count == report['counts']['energy_evaluation']
    for state in (case/'solves').glob('*.json'):
        assert json.loads(state.read_text())['complete'], state
    job = json.loads((case/'job.json').read_text())
    if job['shots'] == 1000:
        readout = json.loads((case/'readout.json').read_text())
        assert sum(readout['counts'].values()) == 1000
    if job['optimizer'] == 'lbfgs':
        assert report['counts'].get('reverse_value_gradient', 0) > 0
    return dict(case=str(case), passed=True, partitions=len(partitions),
                counts=report['counts'], global_points=len(points), attempts=len(report['attempts']))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', type=Path)
    args = parser.parse_args()
    for case in args.cases:
        print(json.dumps(validate(case)), flush=True)
