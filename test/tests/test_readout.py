"""Asymmetric computational states expose swapped qubit numbering."""
from pathlib import Path
import ast
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from readout import protein_probabilities


class ReadoutTests(unittest.TestCase):
    def test_q0_is_rightmost_for_protein_decoder(self):
        self.assertEqual(protein_probabilities({'1000000': 1000}), {'0000001': 1.0})

    def test_probabilities_and_most_likely_state(self):
        probabilities = protein_probabilities({'1010000': 750, '0100001': 250})
        self.assertEqual(probabilities, {'0000101': .75, '1000010': .25})
        self.assertEqual(max(probabilities, key=probabilities.get), '0000101')

    def test_no_data_is_not_a_protein_result(self):
        with self.assertRaises(ValueError):
            protein_probabilities({})

    def test_standalone_readers_retain_distant_samples(self):
        # The old reader searched only one-bit neighbors of the mode and
        # silently discarded 1111111, then renormalized the remaining counts.
        counts = {'0000000': 600, '1111111': 300, '1000000': 100}
        class SampleResult:
            items = lambda self: counts.items()
            count = lambda self, key: counts.get(key, 0)
            most_probable = lambda self: '0000000'
        roots = [ROOT/'sources/independent-vqe',
                 ROOT.parent/'VQE_Docking_cuda-q_independent_linear',
                 ROOT.parent/'VQE_Docking_cuda-q_independent_reverse_linear']
        for root in roots:
            with self.subTest(source=root.name):
                path = root/'Qiskit_VQE/measure.py'
                cls = next(n for n in ast.parse(path.read_text()).body
                           if isinstance(n, ast.ClassDef) and n.name == 'CudaqStateCalculator')
                tree = ast.Module(body=[ast.ImportFrom(module='__future__',
                    names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
                scope = {'cudaq': SimpleNamespace(sample=lambda *args, **kwargs: SampleResult())}
                exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)
                reader = scope['CudaqStateCalculator'](None, shots=1000)
                self.assertEqual(reader.get_probability_distribution([]),
                                 {'0000000': .6, '1111111': .3, '0000001': .1})


if __name__ == '__main__':
    unittest.main()
