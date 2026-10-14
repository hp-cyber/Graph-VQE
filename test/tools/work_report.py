"""Stream the hierarchical ledger into exact scoped counts and time series."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def write_report(case):
    case=Path(case)
    counts=Counter();started=Counter();statuses=Counter();groups=defaultdict(lambda:dict(counts=Counter(),inclusive_seconds=defaultdict(float)))
    pending={};issues=[];attempts={};global_points=0
    with (case/'energy_evaluations.csv').open('w',newline='') as ef, (case/'optimizer_iterations.csv').open('w',newline='') as it, (case/'work_events.jsonl').open() as events:
        columns=['attempt_id','event_id','parent_id','optimizer_call','solve_key','event','role','energy_scope','partition_id','stage_id','epoch','sweep','adapt_step','topology_id','qubits','circuit_qubits','logical_qubits','start_elapsed_seconds','end_elapsed_seconds','seconds','energy','parameter_count','parameters','state_parameter_count','state_parameters','reconstruction_coefficients']
        ew=csv.DictWriter(ef,fieldnames=columns);ew.writeheader()
        iw=csv.DictWriter(it,fieldnames=columns+['iteration','complete']);iw.writeheader()
        for line_no,line in enumerate(events,1):
            try:r=json.loads(line)
            except json.JSONDecodeError:issues.append({'line':line_no,'issue':'incomplete_json_line'});continue
            attempt=r['attempt_id'];a=attempts.setdefault(attempt,dict(first_monotonic_ns=None,last_monotonic_ns=None,status='running'))
            t=r.get('end_monotonic_ns',r.get('monotonic_ns',r.get('start_monotonic_ns')))
            if t is not None:
                a['first_monotonic_ns']=t if a['first_monotonic_ns'] is None else min(t,a['first_monotonic_ns'])
                a['last_monotonic_ns']=t if a['last_monotonic_ns'] is None else max(t,a['last_monotonic_ns'])
            if r['event']=='attempt_finish':a['status']=r['status']
            if r['phase']=='start':
                started[r['event']]+=1
                pending[r['event_id']]={'event':r['event'],'attempt_id':attempt};continue
            if r['phase']=='point':
                if r['event']=='global_energy':global_points+=1
                continue
            pending.pop(r['event_id'],None)
            statuses[(r['event'],r['status'])]+=1
            if r['status']!='complete':continue
            counts[r['event']]+=1
            ctx=r['context']
            group={k:ctx.get(k) for k in ('partition_id','fragment_id','stage_id','epoch','sweep','adapt_step','role','energy_scope','qubits','circuit_qubits')}
            key=json.dumps(group,sort_keys=True)
            groups[key]['counts'][r['event']]+=1;groups[key]['inclusive_seconds'][r['event']]+=r['seconds']
            if r['seconds']<0:issues.append({'line':line_no,'issue':'negative_time'})
            if 'parameters' in r and r.get('parameter_count')!=len(r['parameters']):issues.append({'line':line_no,'issue':'parameter_count_mismatch'})
            row={k:r.get(k,ctx.get(k)) for k in columns}
            for field in ('parameters','state_parameters','reconstruction_coefficients'):
                row[field]=json.dumps(r.get(field))
            if r['event'] in ('energy_evaluation','energy_component','energy_reconstruction'):
                if r.get('energy') is None:issues.append({'line':line_no,'issue':'missing_energy'})
                ew.writerow(row)
            if r['event'] in ('optimizer_iteration','optimizer_iteration_tail'):
                row.update(iteration=r.get('iteration'),complete=r.get('complete'))
                iw.writerow(row)
    # Preserve event order across restarted attempts. elapsed_active_seconds
    # removes downtime; the original per-attempt monotonic clocks stay present.
    offsets={};total=0.
    for attempt,a in attempts.items():
        offsets[attempt]=total
        a['wall_seconds']=(a['last_monotonic_ns']-a['first_monotonic_ns'])/1e9
        total+=a['wall_seconds']
    best=None
    # Stream a second pass: retaining all growing parameter vectors here can
    # otherwise consume gigabytes at the end of a long adaptive run.
    with (case/'global_energy_timeline.jsonl').open('w') as f, (case/'work_events.jsonl').open() as events:
        for line in events:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['event']!='global_energy' or r['phase']!='point':continue
            energy=r['energy'];best=energy if best is None else min(best,energy)
            r.update(elapsed_active_seconds=offsets[r['attempt_id']]+(r['monotonic_ns']-attempts[r['attempt_id']]['first_monotonic_ns'])/1e9,best_energy_so_far=best)
            f.write(json.dumps(r)+'\n')
    reconciliation={}
    for filename,event,expected in (
        ('backend_evaluations.jsonl',None,counts['backend_call']),
        ('optimization_trace.jsonl','energy_evaluation',counts['objective_call']),
        ('optimization_trace.jsonl','iteration',counts['optimizer_iteration'])):
        path=case/filename
        if not path.exists():continue
        observed=0
        with path.open() as legacy:
            for line in legacy:
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if event is None or row.get('event')==event:observed+=1
        key=filename+(':'+event if event else '')
        reconciliation[key]=dict(legacy=observed,ledger=expected,matched=observed==expected)
        # Interrupted attempts can die between the independent file writes.
        # Report this explicitly, without blocking a later valid warm restart.
        if observed!=expected and len(attempts)==1 and next(iter(attempts.values()))['status']=='complete':
            issues.append(dict(issue='counter_mismatch',counter=key,legacy=observed,ledger=expected))
    summary=dict(schema='vqe_work_summary.v1',counts=dict(counts),started_counts=dict(started),
        optimizer_solves_finished= sum(statuses[('optimizer_solve',s)] for s in ('complete','early_stop')),
        reconciliation=reconciliation,
        statuses=[dict(event=k[0],status=k[1],count=v) for k,v in statuses.items()],
        groups=[dict(context=json.loads(k),counts=dict(v['counts']),inclusive_seconds=dict(v['inclusive_seconds'])) for k,v in groups.items()],
        attempts=attempts,active_wall_seconds=total,unclosed_events=pending,
        unclosed_current_attempt={k:v for k,v in pending.items() if v['attempt_id']==list(attempts)[-1]},issues=issues,
        global_energy_points=global_points,
        definitions=dict(energy_evaluation='One scalar Hamiltonian objective evaluation, including initialization and energy probes; nested components are separate.',
            counts='Successfully returned operations only. started_counts includes interrupted attempts; statuses distinguishes normal return, controlled early stop, interruption and failure.',
            pool_gradient='One ADAPT candidate gradient/commutator assessment; not necessarily an energy evaluation.',
            reverse_value_gradient='One joint analytic energy/gradient contraction. Its energy is a gradient-engine value, not another CUDA-Q objective call or a sampled global energy.',
            backend_call='One CUDA-Q API call, potentially returning several Pauli moments; not a global energy count.',
            optimizer_iteration='One completed optimizer callback, excluding initialization and incomplete tail.',
            timing='Durations are inclusive wall time. Parent, child, iteration and evaluation durations overlap and MUST NOT be summed together.',
            timeline='Actual recorded full-H energies/reconstructions in chronological order; no interpolation of local partition energies. Restart downtime excluded only in elapsed_active_seconds.'))
    tmp=case/'work_summary.tmp';tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(case/'work_summary.json')
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path)
    args=parser.parse_args();print(json.dumps(write_report(args.case),indent=2))
