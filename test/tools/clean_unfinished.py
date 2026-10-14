"""Delete incomplete case outputs while preserving inputs and completed cases."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import shutil
from job_state import reserve_case
from record_storage import remove_record_storage


def clean(roots, apply=False):
    report=[]
    with ExitStack() as stack:
        # Exclude launchers throughout inspection and deletion.
        for root in roots:
            paths=list(root.rglob('.campaign.lock'))
            matched=root.parent/'.matched_campaign.lock'
            if matched.exists(): paths.append(matched)
            for path in sorted(set(paths)):
                lock=stack.enter_context(path.open('a'))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for root in roots:
            for manifest in sorted(root.rglob('job.json')):
                case=manifest.parent
                lock,reason=reserve_case(case)
                if reason:
                    report.append(dict(case=str(case),action='preserved',reason=reason));continue
                try:
                    paths=[p for p in case.iterdir() if p.name not in ('job.json','protein_manifest.json','.run.lock')]
                    if not paths: continue
                    if apply:
                        remove_record_storage(case)
                        for path in paths:
                            if path.is_dir() and not path.is_symlink(): shutil.rmtree(path)
                            else: path.unlink()
                    report.append(dict(case=str(case),action='cleaned' if apply else 'would_clean',entries=len(paths)))
                finally: lock.close()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--apply',action='store_true');p.add_argument('--report',type=Path)
    a=p.parse_args();report=clean([r.resolve() for r in a.roots],a.apply)
    if a.report: a.report.write_text(json.dumps(report,indent=2)+'\n')
    from collections import Counter
    print(dict(Counter(r['action'] for r in report)))
