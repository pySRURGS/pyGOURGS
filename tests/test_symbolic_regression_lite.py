import os
import sh
import sys
import types
import time
import pdb
import unittest
sys.path.append('./../pyGOURGS/')
import pyGOURGS.pyGOURGS as pg
import multiprocessing


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
        os.chdir('./examples')
        sh.python3('./symbolic_regression_lite.py', 
                    '-weights', 
                   './weights.csv', 
                   './weights_data.csv', 
                   './test_output.json')

    def test_multiprocessing_performance(self):
        time0 = time.time()        
        sh.python3('./symbolic_regression_lite.py', 
                    '-weights', 
                   './weights.csv',
                   '-num_iters',
                   '1000',
                   '-simplify_solutions',
                   'False',
                   './weights_data.csv', 
                   './test_output.json')
        time1 = time.time()
        diff1 = time1 - time0
        sh.python3('./symbolic_regression_lite.py', 
                    '-weights', 
                   './weights.csv', 
                   '-multiprocessing', 
                   'True',
                   '-simplify_solutions',
                   'False',
                   '-num_iters',
                   '1000',
                   './weights_data.csv',                    
                   './test_output.json')        
        time2 = time.time()
        diff2 = time2 - time1
        print("Single Processing:", diff1)
        print("Multiprocessing:", diff2)
        print("Number cores:", multiprocessing.cpu_count())

if __name__ == '__main__':
    unittest.main()

