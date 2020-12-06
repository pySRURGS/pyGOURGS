import os
import sh
import sys
import types
import unittest
sys.path.append('./../pyGOURGS/')
import pyGOURGS as pg


class TestSymbolicRegression(unittest.TestCase):

    def setUp(self):
        self.pset = pg.PrimitiveSet()
        self.pset.add_operator('add', 2)
        self.pset.add_operator('sub', 1)
        self.pset.add_operator('truediv', 3)
        self.pset.add_operator('mul', 1)
        self.pset.add_variable('x')
        self.pset.add_variable('y')
        self.enum = pg.Enumerator(self.pset)

    def test_uniform_random_global_search(self):
        solns = []
        for soln in self.enum.uniform_random_global_search(10000, 10):
            solns.append(soln)
        self.assertEqual(len(solns), len(list(set(solns))), 10)
        soln = self.enum.uniform_random_global_search_once(10000)
        self.assertEqual(type(soln), str)
        solns = []
        for soln in self.enum.exhaustive_global_search(2,5):
            solns.append(soln)
        self.assertEqual(len(solns), 5)
        func = pg.compile(soln, self.pset)
        self.assertEqual(type(func), types.FunctionType)

    def test_cli(self):
        sh.python3('./symbolic_regression_lite.py', 
                    '-weights', 
                   './weights.csv', 
                   './weights_data.csv', 
                   './test_output.json')

if __name__ == '__main__':
    unittest.main()

