"""Bounded fresh-process restart test for every selected shot/optimizer/pattern."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--shots',type=int,required=True)
args=p.parse_args();rows=[]
for job in json.loads((ROOT/'campaign.json').read_text())['jobs']:
    if job['protein_id']!='3ckz' or job['shots']!=args.shots:continue
    case=ROOT/'validation/resume'/job['key'];case.mkdir(parents=True,exist_ok=True)
    (case/'job.json').write_text(json.dumps(job))
    codes=[]
    for phase in ('interrupt','resume'):
        with (case/f'{phase}.log').open('w') as log:
            r=subprocess.run([sys.executable,str(ROOT/'tools/check_resume_worker.py'),str(case),'--gpu',args.gpu],
                env=dict(os.environ,GRAPH_VQE_MAX_ITER='3'),stdout=log,stderr=subprocess.STDOUT,timeout=180)
        codes.append(r.returncode)
    trace=[json.loads(line) for line in (case/'optimization_trace.jsonl').read_text().splitlines()]
    marker=json.loads((case/'interruption_injected.json').read_text())
    starts=[r for r in trace if r['event']=='start']
    resumed=[r for r in starts if r['resumed'] and r['solve_key']==marker['solve_key'] and r['parameters']==marker['parameters']]
    solves=[json.loads(p.read_text()) for p in (case/'solves').glob('*.json')]
    # Local starts occur only during the first execution; completed local solves are reused.
    local_count=sum(s['phase']=='local' for s in solves)
    passed=codes==[75,0] and len(resumed)==1 and len(starts)==local_count+2 and all(s['complete'] for s in solves)
    row=dict(key=job['key'],codes=codes,local_solves=local_count,exact_global_restart=bool(resumed),passed=passed)
    rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/f'validation/resume_{args.shots}.json').write_text(json.dumps(rows,indent=2))
raise SystemExit(not all(r['passed'] for r in rows))
