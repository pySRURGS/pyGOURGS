import sys
import copy
import random
import numpy
import pdb
from functools import partial
import multiprocessing as mp

import pandas as pandas
import parmap
import tqdm
import sys,os
pygourgs_dir = os.path.join('..', 'pyGOURGS')
if os.path.isfile(os.path.join(pygourgs_dir, 'pyGOURGS.py')) == False:
    msg = "Could not find pyGOURGS.py"
    msg += " Run symbolic_regression.py from within the examples directory."
    raise Exception(msg)
import time
sys.path.append(pygourgs_dir)
import pyGOURGS as pg
import argparse 
from sqlitedict import SqliteDict

def solution_saving_worker(queue, n_items, output_db):
    """
        Takes solutions from the queue of evaluated solutions, 
        then saves them to the database.
    """
    checkpoint = int(n_items/100) + 1
    with SqliteDict(output_db, autocommit=False) as results_dict:
        for j in range(0, n_items):
            [score, soln] = queue.get()
            results_dict[soln] = score
            if j == checkpoint:
                print('  Saving results to db: ' + str(j/n_items))
                results_dict.commit()
        results_dict.commit()

def evalSymbolicRegression(equation_string):
    """
        Evaluates the proposed solution to its numerical value
    """
    # pdb.set_trace()
    # QUESTION: if we're predicting y using x, only x should be considered in the equation_string?
    # for x_value in x_column:
        # evaluteEquation = lambda x: equation_string
        # return evaluteEquation(1st value of x from dataframe)
    # QUESTION: are we evaluating only one value of x from the dataframe, or somehow all the values of x? In other words, what specifically is the return value of this method?
    # 
    # value = eval(equation_string)
    # figure out when eval(x) x has already been created
    # better to have dict like object instead of having multiple variable.
    # dict keys names of varialbes, values the arrays of data
    # TODO
    # this logic is not correct and needs to be written. Take this quartic polynomial csv file
    # and predict y from x. 
    # the user defined variables and fitting parameters will not be in this 
    # scope
    #   If user gives us a CSV with header x,y,z we want to ensure that 
    #   whatever these names are, we can access their arrays within this function
    
    # TODO we need to ensure that fitting parameters are recognized and a suitable
    # nonlinear optimization package is used to find optimal values for these 
    # fitting parameters. We can try Levenburg-Marquardt algorithm via the LMFIT 
    # software https://lmfit.github.io/lmfit-py/ as was done in pySRURGS
    # raise Exception("fix this")
    return 1

# TODO
# def simplify_equation_string(equation_string, ?more args):
#   largely copy-paste from simplify_equation_string in pySRURGS.py
#

def main_rando(seed, enum, max_tree_complx):
    """
        evaluates a randomly generated solution
    """
    soln = enum.uniform_random_global_search_once(max_tree_complx, seed=seed)
    pdb.set_trace()
    score = evalSymbolicRegression(soln)    
    return score, soln

def main_rando_queue(seed, enum, max_tree_complx, queue):
    """
        evaluates a randomly generated solution
        and puts it in the queue 
        used for multiprocessing
    """
    soln = enum.uniform_random_global_search_once(max_tree_complx, seed=seed)
    score = evalSymbolicRegression(soln)    
    queue.put([score, soln])

def main(soln):
    """
        evaluates a proposed solution
    """
    score = evalSymbolicRegression(soln)    
    return score, soln

def main_queue(soln, queue):
    """
        evaluates a proposed solution
        and puts the solution in the queue
        used for multiprocessing
    """
    score = evalSymbolicRegression(soln)    
    queue.put([score, soln])

def str2bool(v):
    '''
        This helper function takes various ways of specifying True/False and 
        unifies them. Used in the command line interface.
    '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='symbolic_regression.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-csv_path", help="An absolute filepath of the csv that will be parsed.")
    parser.add_argument("-operators", nargs='+', help="Operators used to create the solution. Permitted:add,sub,mul,div,pow,sin,cos,tan,exp,log,sinh,cosh,tanh.", default=["add", "sub", "mul", "div", "pow"])
    parser.add_argument("output_db", help="An absolute filepath where we save results to a SQLite database. Include the filename. Extension is typically '.db'")
    parser.add_argument("-num_trees", help="pyGOURGS iterates through all the possible trees using an enumeration scheme. This argument specifies the number of trees to which we restrict our search.", type=int, default=10000)
    parser.add_argument("-num_iters", help="An integer specifying the number of search strategies to be attempted in this run", type=int, default=1000)
    parser.add_argument("-freq_print", help="An integer specifying how many strategies should be attempted before printing current job status", type=int, default=10)
    parser.add_argument("-deterministic", help="should algorithm be run in deterministic manner?", type=str2bool, default=False)
    parser.add_argument("-exhaustive", help="should algorithm be run in exhaustive/brute-force mode? This can run forever if you are not careful.", type=str2bool, default=False)
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
    csv_path = arguments.csv_path
    inputted_operators = arguments.operators

    dataframe = pandas.read_csv(csv_path)
    operator_arity = {"add": 2,
                      "sub": 2,
                      "div": 2,
                      "mul": 2,
                      "pow": 2,
                      "exp": 1,
                      "sin": 1,
                      "sinh": 1,
                      "cos": 1,
                      "cosh": 1,
                      "tan": 1,
                      "tanh": 1}  # A dict mapping operators to their arity

    pset = pg.PrimitiveSet()
    for operator in inputted_operators:
        assert operator in operator_arity.keys()  # Make sure the operators entered are allowed
        pset.add_operator(operator, operator_arity[operator])
    for variable in dataframe.columns:
        pset.add_variable(variable)
    enum = pg.Enumerator(pset)
    
    if deterministic == False:
        deterministic = None
    max_score = 0
    iter = 0
    manager = mp.Manager()
    queue = manager.Queue()
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
            runner = mp.Process(target=solution_saving_worker, 
                             args=(queue, num_solns, output_db))
            runner.start()
            for soln in enum.exhaustive_global_search(
                                                 maximum_tree_complexity_index):
                jobs.append(soln)
                iter = iter + 1
                print('\r' + "Progress: " + str(iter/num_solns), end='')
            results = parmap.map(main_queue, jobs, queue=queue,
                                 pm_pbar=True, pm_chunksize=3)
            runner.join()
        elif multiproc == False:
            for soln in enum.exhaustive_global_search(
                                                 maximum_tree_complexity_index):
                score = main(soln)[0]
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
        num_solns = n_iters
        if multiproc == True:
            current_time = int(time.time())
            seeds = numpy.np.arange(0, n_iters)
            seeds = seeds*current_time
            seeds = seeds.tolist()
            runner = mp.Process(target=solution_saving_worker, 
                             args=(queue, num_solns, output_db))
            runner.start()
            results = parmap.map(main_rando_queue, seeds, enum=enum, 
                                 max_tree_complx=maximum_tree_complexity_index, 
                                 queue=queue, pm_pbar=True, pm_chunksize=3)
            runner.join()
        elif multiproc == False:
            for soln in enum.uniform_random_global_search(
                                                  maximum_tree_complexity_index, 
                                                   n_iters, seed=deterministic):
                score = main(soln)[0]
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
