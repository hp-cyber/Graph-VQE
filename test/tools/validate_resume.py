"""Bounded fresh-process restart test for every selected shot/optimizer/pattern."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--shots',type=int,required=True)
p.add_argument('--phase',choices=('local','global'),default='global')
p.add_argument('--output',type=Path,default=ROOT/'validation/resume')
args=p.parse_args();rows=[]
for job in json.loads((ROOT/'campaign.json').read_text())['jobs']:
    if job['protein_id']!='3ckz' or job['shots']!=args.shots:continue
    case=args.output/args.phase/job['key'];case.mkdir(parents=True,exist_ok=True)
    (case/'job.json').write_text(json.dumps(job))
    codes=[]
    for phase in ('interrupt','resume'):
        with (case/f'{phase}.log').open('w') as log:
            r=subprocess.run([sys.executable,str(ROOT/'tools/check_resume_worker.py'),str(case),'--gpu',args.gpu],
                env=dict(os.environ,GRAPH_VQE_MAX_ITER='3',GRAPH_RESUME_PHASE=args.phase),stdout=log,stderr=subprocess.STDOUT)
        codes.append(r.returncode)
    trace=[json.loads(line) for line in (case/'optimization_trace.jsonl').read_text().splitlines()]
    marker=json.loads((case/'interruption_injected.json').read_text())
    starts=[r for r in trace if r['event']=='start']
    resumed=[r for r in starts if r['resumed'] and r['solve_key']==marker['solve_key'] and r['parameters']==marker['parameters']]
    solves=[json.loads(p.read_text()) for p in (case/'solves').glob('*.json')]
    # Every actual optimizer solve starts once, except the deliberately interrupted one.
    local_count=sum(s['phase']=='local' for s in solves)
    keys={s['solve_key'] for s in starts}
    passed=codes==[75,0] and len(resumed)==1 and len(starts)==len(keys)+1 and all(s['complete'] for s in solves)
    from validate_work_records import validate
    records=validate(case)
    row=dict(key=job['key'],phase=args.phase,codes=codes,local_solves=local_count,
             exact_parameter_restart=bool(resumed),work_records=records,passed=passed)
    rows.append(row);print(json.dumps(row),flush=True)
    (args.output/f'resume_{args.phase}_{args.shots}.json').write_text(json.dumps(rows,indent=2))
raise SystemExit(not all(r['passed'] for r in rows))
