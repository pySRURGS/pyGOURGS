import sys
import copy
import random
import numpy
import pdb
from functools import partial
import multiprocessing as mp
import numpy
import pandas as pandas
import parmap
import numexpr as ne
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

def evalSymbolicRegression(equation_string, data, mode='residual'):
    """
        Evaluates the proposed solution to its numerical value
    """
    # TODO need to add weights, fitting parameters, and simplify equations
    data_dict = data.get_data_dict()
    independent_vars_vector, x_label = data.get_independent_data()  # x vector (or more)
    dependent_var_vector, y_label = data.get_dependent_data()  # y vector
    y_predicted = ne.evaluate(equation_string, local_dict=data_dict)  # evaluates the equation using the dict to retrieve
    # the values of the variables

    y_actual = dependent_var_vector

    if mode == 'residual':
        residual = y_actual - y_predicted
        # if data._data_weights is not None:  # TODO
        #     residual = numpy.multiply(residual, data._data_weights)
        print(residual)
        output = float(numpy.sum(residual**2))  # residual sum of squares
    elif mode == 'y_calc':
        output = y_predicted
    elif type(mode) == dict:  # QUESTION: what is this?
        df = mode
        y_value = eval(equation_string)
        output = y_value
    if numpy.size(output) == 1:
        # if model only has parameters and no data variables, we can have a
        # situation where output is a single constant
        output = numpy.resize(output, numpy.size(independent_vars_vector))

    return output
    
    # TODO we need to ensure that fitting parameters are recognized and a suitable
    # nonlinear optimization package is used to find optimal values for these 
    # fitting parameters. We can try Levenburg-Marquardt algorithm via the LMFIT 
    # software https://lmfit.github.io/lmfit-py/ as was done in pySRURGS
    # raise Exception("fix this")

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

def main(soln, data):
    """
        evaluates a proposed solution
    """
    score = evalSymbolicRegression(soln, data)
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

def customAssert(condition, action):
    """
    Allows us to create custom assertions
    """
    if not condition: raise action

class DataHelper:
    def __init__(self, dataframe, variables):
        self.dataframe = dataframe
        self.variables = variables
        self.header_labels = []
        for label in dataframe.columns[:]:
            self.header_labels.append(label)

        # self.y_label = y_label

    def get_data_dict(self):
        '''
            Creates a dictionary object which houses the values in the dataset
            CSV. The variable names in the CSV become keys in this data_dict
            dictionary.
        '''
        data_dict = dict()
        for label in self.header_labels:
            data_dict[label] = numpy.array(self.dataframe[label].values).astype(float)
            # check_for_nans(data_dict[label])
        return data_dict

    def get_independent_data(self):
        '''
            Loads all data in self._dataframe except the rightmost column
            Example: f(x, y) = z, get_independent_data would return x, y
        '''
        dataframe = self.dataframe
        header_labels = self.header_labels
        features = dataframe.iloc[:, :-1]
        features = numpy.array(features)
        labels = header_labels[:-1]
        # properties = dict() TODO?
        # for label in labels:
        #     properties.update(get_properties(dataframe[label], label))
        return (features, labels)

    def get_dependent_data(self):
        '''
            Loads only the rightmost column from self._dataframe
            Example: f(x, y) = z, get_dependent_data would return z
        '''
        dataframe = self.dataframe
        header_labels = self.header_labels
        feature = dataframe.iloc[:, -1]
        feature = numpy.array(feature)
        label = header_labels[-1]
        # properties = get_properties(dataframe[label], label)
        return (feature, label)

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

    dataframe = pandas.read_csv(csv_path)  # maybe create a new class
    variables = []
    for variable in dataframe.columns[:-1]:
        variables.append(variable)
    data = DataHelper(dataframe, variables)
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
        customAssert(operator in operator_arity.keys(), Exception("Invalid operator entered {}. Permitted operators: add,sub,mul,div,pow,sin,cos,tan,exp,log,sinh,cosh,tanh.".format(operator)))
        pset.add_operator(operator, operator_arity[operator])
    for variable in data.variables:  # TODO: add local varialbes and method that returns the independent vars labels
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
                score = main(soln, data)[0]
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
