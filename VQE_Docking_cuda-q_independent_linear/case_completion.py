"""Completion checks, including outputs produced before completion markers existed."""
import json
import math
import os
from pathlib import Path
import time
import uuid


def _valid_outputs(root, metadata, sequence, protein_id, strategy):
    if metadata.get('strategy') != strategy or metadata.get('sequence', sequence) != sequence:
        return False
    count = int(metadata['iterations'])
    energies = (root/'System_Energy'/f'energy_list_{protein_id}.txt').read_text().splitlines()
    if count < 1 or len(energies) != count or not all(math.isfinite(float(e)) for e in energies):
        return False
    probabilities = (root/'Prob_distribution'/f'prob_distribution_{protein_id}.txt').read_text().splitlines()
    if not probabilities:
        return False
    for line in probabilities:
        state, value = line.rsplit(':', 1)
        probability = float(value)
        if not state.strip() or set(state.strip()) - {'0', '1'} or not math.isfinite(probability) or not 0 <= probability <= 1:
            return False
    # IndependentSubspaceVQE retains the six best global evaluations.
    top_count = int(metadata.get('top_results_count', min(6, count)))
    if not 1 <= top_count <= 6:
        return False
    names = [f'{protein_id}_{strategy}'] + [f'{protein_id}_{strategy}_top_{i}' for i in range(1, top_count+1)]
    for name in names:
        lines = (root/'Structures'/f'{name}.xyz').read_text().splitlines()
        atoms = [line.split() for line in lines[2:] if line.strip()]
        if int(lines[0]) != len(sequence) or len(atoms) != len(sequence):
            return False
        if ''.join(atom[0] for atom in atoms) != sequence:
            return False
        if any(len(atom) != 4 or not all(math.isfinite(float(v)) for v in atom[1:]) for atom in atoms):
            return False
    return True


def is_complete(root, sequence, protein_id, strategy):
    root = Path(root)
    try:
        metadata = json.loads((root/'stage_summary.json').read_text())
        marker = root/'COMPLETED.json'
        if marker.exists():
            record = json.loads(marker.read_text())
            if any(record.get(k) != v for k, v in dict(
                    status='complete', sequence=sequence, protein_id=protein_id, strategy=strategy).items()):
                return False
        elif metadata.get('completion_protocol'):
            # New runs write their marker only after all final readouts/exports.
            return False
        return _valid_outputs(root, metadata, sequence, protein_id, strategy)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False


def mark_complete(root, sequence, protein_id, strategy):
    root = Path(root)
    try:
        metadata = json.loads((root/'stage_summary.json').read_text())
        valid = _valid_outputs(root, metadata, sequence, protein_id, strategy)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        valid = False
    if not valid:
        raise RuntimeError(f'Cannot mark incomplete outputs as finished: {root}')
    record = dict(status='complete', sequence=sequence, protein_id=protein_id,
                  strategy=strategy, finished=time.time())
    temporary = root/f'.COMPLETED.{uuid.uuid4().hex}.tmp'
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    os.replace(temporary, root/'COMPLETED.json')
