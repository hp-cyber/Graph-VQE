from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from graph_resources import allocation_bound


class GraphResourceTests(unittest.TestCase):
    def test_shallow_nearest_neighbor_97_qubit_cap_is_small(self):
        for reverse in (False,True):
            edges=[(i,i+1) for i in range(96)]
            if reverse:edges.reverse()
            result=allocation_bound(97,edges*4,16,1024**3)
            self.assertTrue(result['passed'])
            self.assertEqual(result['max_allocated_bond'],16)

    def test_small_graph_fits_and_dense_large_graph_stops_before_allocation(self):
        edges=[(a,b) for a in range(7) for b in range(a+1,7)]
        small=allocation_bound(7,edges*4,8,1024**3)
        self.assertTrue(small['passed'])
        large=allocation_bound(97,((a,b) for a in range(97) for b in range(a+1,97)),2**31-1,8*1024**3)
        self.assertFalse(large['passed'])
        self.assertGreater(large['estimated_peak_bytes'],8*1024**3)
        self.assertLess(large['routed_two_qubit_gates'],10000)


if __name__=='__main__':unittest.main()
