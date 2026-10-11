"""CPU-only regression checks for finished-case skipping and partial outputs."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from case_completion import is_complete, mark_complete

PROJECT = Path(__file__).resolve().parent
STRATEGY = 'subspace_hamiltonian_' + ('reverse_linear' if 'reverse_linear' in PROJECT.name else 'linear')


def fixture(root, modern=False):
    for name in ('System_Energy', 'Prob_distribution', 'Structures'):
        (root/name).mkdir(parents=True, exist_ok=True)
    metadata = dict(strategy=STRATEGY, iterations=2)
    if modern:
        metadata.update(completion_protocol=1, sequence='VK', protein_id='p', top_results_count=2)
    (root/'stage_summary.json').write_text(json.dumps(metadata))
    (root/'System_Energy/energy_list_p.txt').write_text('1.0\n0.5\n')
    (root/'Prob_distribution/prob_distribution_p.txt').write_text('00: 1.0\n')
    for suffix in ('', '_top_1', '_top_2'):
        (root/'Structures'/f'p_{STRATEGY}{suffix}.xyz').write_text('2\n\nV 0 0 0\nK 1 0 0\n')


def load_function(name, namespace):
    tree = ast.parse((PROJECT/'Main.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(PROJECT/'Main.py'), 'exec'), namespace)
    return namespace[name]


class CompletionTests(unittest.TestCase):
    def test_legacy_complete_and_missing_final_structure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);fixture(root)
            self.assertTrue(is_complete(root, 'VK', 'p', STRATEGY))
            (root/'Structures'/f'p_{STRATEGY}_top_2.xyz').unlink()
            self.assertFalse(is_complete(root, 'VK', 'p', STRATEGY))
            with self.assertRaises(RuntimeError):mark_complete(root, 'VK', 'p', STRATEGY)

    def test_new_run_requires_atomic_final_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);fixture(root, modern=True)
            self.assertFalse(is_complete(root, 'VK', 'p', STRATEGY))
            mark_complete(root, 'VK', 'p', STRATEGY)
            self.assertTrue(is_complete(root, 'VK', 'p', STRATEGY))
            (root/'COMPLETED.json').write_text('{')
            self.assertFalse(is_complete(root, 'VK', 'p', STRATEGY))

    def test_wrong_sequence_or_strategy_and_corrupt_energy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);fixture(root)
            self.assertFalse(is_complete(root, 'KV', 'p', STRATEGY))
            self.assertFalse(is_complete(root, 'VK', 'p', 'different_strategy'))
            (root/'System_Energy/energy_list_p.txt').write_text('1.0\nnan\n')
            self.assertFalse(is_complete(root, 'VK', 'p', STRATEGY))

    def test_partial_timing_only_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);(root/'System_Energy').mkdir()
            (root/'System_Energy/energy_evaluation_timings.csv').write_text('energy\n2\n')
            self.assertFalse(is_complete(root, 'VK', 'p', STRATEGY))

    def test_prediction_skips_before_optimizer_and_readout(self):
        namespace = dict(os=os, is_complete=is_complete, mark_complete=mark_complete,
                         num_gpus=1, USE_PARALLEL_BLOCKS=False)
        for name in ('Peptide', 'MiyazawaJerniganInteraction', 'PenaltyParameters',
                     'ProteinFoldingProblem', 'qiskit_sparsepauliop_to_cudaq_spin',
                     'IndependentSubspaceVQE', 'CUDAQVQE', 'CudaqStateCalculator', 'configure_gpu_groups'):
            namespace[name] = Mock(name=name)
        predict = load_function('predict_protein_structure', namespace)
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                fixture(Path('Result/process_data')/STRATEGY/'p')
                self.assertEqual(predict('VK', 'p', 'cudaq', None), [])
                for name in ('IndependentSubspaceVQE', 'CUDAQVQE', 'CudaqStateCalculator', 'configure_gpu_groups'):
                    namespace[name].assert_not_called()
            finally:os.chdir(original)

    def test_main_preserves_existing_time_log(self):
        namespace = dict(os=os, predict_protein_structure=Mock(return_value=[]))
        main = load_function('main', namespace)
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                log = Path('execution_time_log.txt');log.write_text('existing run\n')
                main()
                self.assertEqual(log.read_text(), 'existing run\n')
            finally:os.chdir(original)


if __name__ == '__main__':
    unittest.main()
