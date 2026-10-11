"""Test-only interruption after a persisted objective in the global solve."""
from pathlib import Path
import json
from optimization_trace import Journal

original=Journal.save


def saved(self,**data):
    original(self,**data)
    if 'x' not in data or 'energy' not in data:return
    path=Path(self.identity['namespace'])
    state=json.loads(path.read_text())
    marker=Path('interruption_injected.json')
    if state['phase']=='global' and not marker.exists():
        marker.write_text(json.dumps(dict(solve_key=self.path.stem,parameters=data['x'])))
        raise SystemExit(75)


if __name__=='__main__':
    Journal.save=saved
    from run_case import main
    main()
