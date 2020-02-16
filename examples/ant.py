# This code is derived from the DEAP project.
# See https://github.com/DEAP/deap/blob/master/examples/gp/ant.py
import sys
import copy
import random
import numpy
import pdb
from functools import partial
import multiprocessing
import parmap
import tqdm
import sys,os
sys.path.append(os.path.join('..', 'pyGOURGS'))
import pyGOURGS as pg
import argparse 

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2): 
    return partial(progn,out1,out2)

def prog3(out1, out2, out3):     
    return partial(progn,out1,out2,out3)

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()

class AntSimulator(object):
    direction = ["north","east","south","west"]
    dir_row = [1, 0, -1, 0]
    dir_col = [0, 1, 0, -1]
    
    def __init__(self, max_moves):
        self.max_moves = max_moves
        self.moves = 0
        self.eaten = 0
        self.routine = None
        
    def _reset(self):
        self.row = self.row_start 
        self.col = self.col_start 
        self.dir = 1
        self.moves = 0  
        self.eaten = 0
        self.matrix_exc = copy.deepcopy(self.matrix)

    @property
    def position(self):
        return (self.row, self.col, self.direction[self.dir])
            
    def turn_left(self): 
        if self.moves < self.max_moves:
            self.moves += 1
            self.dir = (self.dir - 1) % 4

    def turn_right(self):
        if self.moves < self.max_moves:
            self.moves += 1    
            self.dir = (self.dir + 1) % 4
        
    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.row = (self.row + self.dir_row[self.dir]) % self.matrix_row
            self.col = (self.col + self.dir_col[self.dir]) % self.matrix_col
            if self.matrix_exc[self.row][self.col] == "food":
                self.eaten += 1
            self.matrix_exc[self.row][self.col] = "passed"

    def sense_food(self):
        ahead_row = (self.row + self.dir_row[self.dir]) % self.matrix_row
        ahead_col = (self.col + self.dir_col[self.dir]) % self.matrix_col        
        return self.matrix_exc[ahead_row][ahead_col] == "food"
   
    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food, out1, out2)
   
    def run(self,routine):
        self._reset()
        while self.moves < self.max_moves:
            routine()
    
    def parse_matrix(self, matrix):
        self.matrix = list()
        for i, line in enumerate(matrix):
            self.matrix.append(list())
            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("food")
                elif col == ".":
                    self.matrix[-1].append("empty")
                elif col == "S":
                    self.matrix[-1].append("empty")
                    self.row_start = self.row = i
                    self.col_start = self.col = j
                    self.dir = 1
        self.matrix_row = len(self.matrix)
        self.matrix_col = len(self.matrix[0])
        self.matrix_exc = copy.deepcopy(self.matrix)

pset = pg.PrimitiveSet()
pset.add_operator("ant.if_food_ahead", 2)
pset.add_operator("prog2", 2)
pset.add_operator("prog3", 3)
pset.add_variable("ant.move_forward()")
pset.add_variable("ant.turn_left()")
pset.add_variable("ant.turn_right()")
enum = pg.Enumerator(pset)

def evalArtificialAnt(search_strategy_string):
    # Transform the tree expression to Python code
    routine = eval('lambda : ' + search_strategy_string)
    # Run the generated routine
    ant.run(routine)
    return ant.eaten

def main(soln, output_db):    
    score = evalArtificialAnt(soln)    
    return score, soln

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

ant = AntSimulator(600)
with open("./johnmuir_trail.txt") as trail_file:
    ant.parse_matrix(trail_file)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ant.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_db", help="An absolute filepath where we save results to a SQLite database. Include the filename. Extension is typically '.db'")
    parser.add_argument("-num_trees", help="pyGOURGS iterates through all the possible trees using an enumeration scheme. This argument specifies the number of trees to which we restrict our search.", type=int, default=10000)
    parser.add_argument("-num_iters", help="An integer specifying the number of search strategies to be attempted in this run", type=int, default=1000)
    parser.add_argument("-freq_print", help="An integer specifying how many strategies should be attempted before printing current job status", type=int, default=10)
    parser.add_argument("-deterministic", help="should algorithm be run in deterministic manner?", type=str2bool, default=False)
    parser.add_argument("-exhaustive", help="should algorithm be run in exhaustive/brute-force mode? This can run forever if you are not careful.", type=bool, default=False)
    parser.add_argument("-multiprocessing", help="should algorithm be run in multiprocessing mode?", type=str2bool, default=False)
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    arguments = parser.parse_args()
    output_db = arguments.output_db
    n_iters = arguments.num_iters
    maximum_tree_complexity_index = arguments.num_trees
    frequency_printing = arguments.freq_print
    deterministic = arguments.deterministic
    exhaustive = arguments.exhaustive
    multiproc = arguments.multiprocessing
    max_score = 0
    iter = 0    
    if exhaustive == True:
        _, weights = enum.calculate_Q(maximum_tree_complexity_index)
        num_solns = int(numpy.sum(weights))
        txt = input("The number of equations to be considered is " + 
                    str(num_solns) + ", do you want to proceed?" + 
                    " If yes, press 'c' then 'enter'.")
        if txt != 'c':
            print("You input: " + txt + ", exiting...")
            exit(1)
        if multiproc == True:
            jobs = []
            for soln in enum.exhaustive_global_search(
                                                 maximum_tree_complexity_index):
                jobs.append(soln)
                iter = iter + 1
                print('\r' + "Progress: " + str(iter/num_solns), end='')
            results = parmap.map(main, jobs, output_db=output_db, 
                                 pm_pbar=True, pm_chunksize=3)
            for (score, soln) in results:
                pg.save_result_to_db(output_db, score, soln)
            iter = 0 
            for result in results:
                iter = iter + 1
                score = result[0]
                if score > max_score:
                    max_score = score
                if iter % frequency_printing == 0:
                    print("best score of this run:" + str(max_score), 
                          'iteration:'+ str(iter), end='\r')
        elif multiproc == False:
            for soln in enum.exhaustive_global_search(
                                                 maximum_tree_complexity_index):
                score = main(soln, output_db)[0]
                pg.save_result_to_db(output_db, score, soln)
                iter = iter + 1
                if score > max_score:
                    max_score = score
                if iter % frequency_printing == 0:
                    print("best score of this run:" + str(max_score), 
                          'iteration:'+ str(iter), end='\r')
        else:
            raise Exception("Invalid value multiproc must be true/false")
    elif exhaustive == False:
        if multiproc == True:
            jobs = []
            for soln in enum.uniform_random_global_search(
                                         maximum_tree_complexity_index, n_iters, 
                                                   deterministic=deterministic):
                jobs.append(soln)
            results = parmap.map(main, jobs, output_db=output_db, 
                                 pm_pbar=True, pm_chunksize=3)
            for (score, soln) in results:
                pg.save_result_to_db(output_db, score, soln)
            for result in results:
                iter = iter + 1
                score = result[0]
                if score > max_score:
                    max_score = score
                if iter % frequency_printing == 0:
                    print("best score of this run:" + str(max_score), 
                          'iteration:'+ str(iter), end='\r')
        elif multiproc == False:
            for soln in enum.uniform_random_global_search(
                                                  maximum_tree_complexity_index, 
                                          n_iters, deterministic=deterministic):
                score = main(soln, output_db)[0]
                pg.save_result_to_db(output_db, score, soln)
                iter = iter + 1
                if score > max_score:
                    max_score = score
                if iter % frequency_printing == 0:
                    print("best score of this run:" + str(max_score), 
                          'iteration:'+ str(iter), end='\r')
        else:
            raise Exception("Invalid multiproc, must be true/false")    
    else:
        raise Exception("Invalid value for exhaustive")
    pg.ResultList(output_db)
