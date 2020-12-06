# Symbolic regression using PyGOURGS
# Sohrab Towfighi BASc MD, Copyright 2020


import sys
import csv
import copy
import random
import numpy as np
import pdb
from functools import partial
import multiprocessing as mp
import pandas
import parmap
import tqdm
import lmfit
import sys,os
import sympy
from operator import itemgetter
from math_funcs import (sympy_Sub, sympy_Div, sin, cos, tan, exp, log, sinh,
                        cosh, tanh, sum, add, sub, mul, div, pow, sqrt)
from sympy import simplify, sympify, Symbol
pygourgs_dir = os.path.join('..', 'pyGOURGS')
if os.path.isfile(os.path.join(pygourgs_dir, 'pyGOURGS.py')) == False:
    msg = "Could not find pyGOURGS.py"
    msg += " Run symbolic_regression.py from within the examples directory."
    raise Exception(msg)
import time
sys.path.append(pygourgs_dir)
import pyGOURGS as pg
import argparse 
import json
import sklearn.metrics as metrics


def regression_results(y_true, y_pred):
    '''
        Returns a dictionary object with regression performance metrics 
        housed internally. The goal of EvalSymbolicRegression can be 
        set to any of the keys of this dictionary.
    '''
    try:
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) 
        mse = metrics.mean_squared_error(y_true, y_pred) 
        mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
        median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
    except ValueError: # infinity, nans, etc
        explained_variance = 0
        mean_absolute_error = np.inf
        mse = np.inf
        mean_squared_log_error = np.inf
        median_absolute_error = np.inf
        r2 = np.inf
    results_dict = {'explained_variance': explained_variance,
                    'mean_absolute_error': mean_absolute_error,
                    'mse': mse,
                    'mean_squared_log_error': mean_squared_log_error,
                    'median_absolute_error': median_absolute_error,
                    'r2': r2}
    return results_dict


class HallOfFame(object):
    def __init__(self, max_len, path_to_json):
        self._manager = mp.Manager()
        self._hof  = self._manager.list()
        self._max_len = max_len
        self._path_to_json = path_to_json
    def insert(self, item, fitness, parameters, metrics, mode='min'):
        if mode not in ['min', 'max', 'zero']:
            raise Exception("Invalid mode")
        try:
            float(fitness)
        except:
            raise Exception("Fitness cannot be cast as float, invalid type")
        # only add the item if it does not exist in the hall of fame
        getitems = itemgetter(0)
        all_items = list(map(getitems, self._hof))
        if item in all_items:
            return
        else:
            pass # continue to append the item to the hall of fame
        self._hof.append([str(item), 
                          float(fitness), 
                          str(parameters), 
                          str(metrics)])
        self._hof = sorted(self._hof, key=itemgetter(1))
        long_len = len(self._hof)        
        for i in range(0, long_len - self._max_len):
            del self._hof[-1]
    def save_to_json(self):
        jsonStr = json.dumps(list(self._hof))
        with open(self._path_to_json, 'w+') as myfile:
            myfile.write(jsonStr)
    def print_best(self):
        best_soln = self._hof[0]
        print("###############################################################")
        print('Best equation:', best_soln[0], '\n')
        print('Fitting parameter values:', best_soln[2], '\n')
        print('Goodness of fit metrics:', best_soln[3])
        print("###############################################################")

# GLOBAL VARIABLES
fitting_param_prefix = "p" #'begin_fitting_param_'
fitting_param_suffix = "" #'_end_fitting_param'
variable_prefix = '' #'begin_variable_'
variable_suffix = '' #'_end_variable'


def has_nans(X):
    if np.any(np.isnan(X)):
        return True
    else:
        return False


def check_for_nans(X):
    if has_nans(X):
        raise Exception("Has NaNs")


def make_parameter_name(par):
    """
    Converts a fitting parameter name to pyGOURGS safe parameter name. Prevents
    string manipulations on parameter names from affecting function names.

    Parameters
    ----------
    par : string
        A variable name.

    Returns
    -------
    par_name: string
        `par` wrapped in the pyGOURGS parameter prefix and suffix.
    """
    par_name = fitting_param_prefix + str(par) + fitting_param_suffix
    return par_name
    

def create_parameter_list(m):
    """
    Creates a list of all the fitting parameter names.

    Parameters
    ----------
    m : int
        The number of fitting parameters in the symbolic regression problem

    Returns
    -------
    my_pars: list
        A list with fitting parameter names as elements.
    """
    my_pars = []
    lmfit_parameter = lmfit.Parameters()    
    for i in range(0, m):
        my_pars.append(make_parameter_name(str(i)))
        lmfit_parameter['p'+str(i)] = lmfit.Parameter(name='p'+str(i), value=1)
    return my_pars, lmfit_parameter


def make_variable_name(var):
    """
    Converts a variable name to pyGOURGS safe variable names. Prevents string
    manipulations on variable names from affecting function names.

    Parameters
    ----------
    var : string
        A variable name.

    Returns
    -------
    var_name: string
        `var` wrapped in the pyGOURGS variable prefix and suffix.
    """
    var_name = variable_prefix + str(var) + variable_suffix
    return var_name
    

def create_variable_list(m):
    """
    Creates a list of all the variable names.

    Parameters
    ----------
    m : string (1) or int (2)
        (1) Absolute or relative path to a CSV file with a header
        (2) The number of independent variables in the dataset

    Returns
    -------
    my_vars: list
        A list with dataset variable names as elements.
    """
    if type(m) == str:
        my_vars = pandas.read_csv(m).keys()[:-1].tolist()
        my_vars = [make_variable_name(x) for x in my_vars]
    if type(m) == int:
        my_vars = []
        for i in range(0, m):
            my_vars.append(make_variable_name('x' + str(i)))
    return my_vars


def is_csv_valid(filepath, check_header=False):
    try:
        with open(filepath, 'r') as csv_file:
            dialect = csv.Sniffer().sniff(csv_file.read(2048))
    except Exception as e:
        print("Error encountering while reading: ", filepath)
        print(e)
        exit(2)
    if check_header == True:        
        with open(filepath, 'r') as csv_file:
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(csv_file.read(2048))
        if has_header == False:
            print("File which must have header is missing header: ", filepath)
            exit(2)


def create_fitting_parameters(max_params, param_values=None):
    """
    Creates the lmfit.Parameters object based on the number of fitting 
    parameters permitted in this symbolic regression problem.

    Parameters
    ----------
    max_params: int
        The maximum number of fitting parameters. Same as `max_num_fit_params`.

    param_values: None OR (numpy.array of length max_params)
        Specifies the values of the fitting parameters. If none, will default
        to an array of ones, which are to be optimized later.

    Returns
    -------
    params: lmfit.Parameters
        Fitting parameter names specified as ['p' + str(integer) for integer
        in range(0, max_params)]
    """
    params = lmfit.Parameters()
    for int_param in range(0, max_params):
        param_name = 'p' + str(int_param)
        param_init_value = np.float(1)
        params.add(param_name, param_init_value)
    if param_values is not None:
        for int_param in range(0, max_params):
            param_name = 'p' + str(int_param)
            params[param_name].value = param_values[int_param]
    return params


def check_validity_suggested_functions(suggested_funcs, arity):
    '''
    Takes a list of suggested functions to use in the search space and checks
    that they are valid.

    Parameters
    ----------
    suggested_funcs: list
        A list of strings.
        In case of `arity==1`, permitted values are ['sin','cos','tan','exp',
                                                     'log','tanh','sinh','cosh',
                                                     None]
        In case of `arity==2`, permitted values are ['add','sub','mul','div',
                                                     'pow', None]

    Returns
    -------
    suggested_funcs: list

    Raises
    ------
    Exception, if any of the suggested funcs is not in the permitted list
    '''
    valid_funcs_arity_1 = ['sin', 'cos', 'tan', 'exp', 'log', 'tanh', 'sinh', 
                           'cosh', None]
    valid_funcs_arity_2 = ['add', 'sub', 'mul', 'div', 'pow', None]
    if arity == 1:
        if suggested_funcs != [',']:
            for func in suggested_funcs:
                if func not in valid_funcs_arity_1:
                    msg = "Your suggested function of arity 1: " + func
                    msg += " is not in the list of valid functions"
                    msg += " " + str(valid_funcs_arity_1)
                    raise Exception(msg)
        else:
            suggested_funcs = []
    elif arity == 2:
        for func in suggested_funcs:
            if func not in valid_funcs_arity_2:
                msg = "Your suggested function of arity 2: " + func
                msg += " is not in the list of valid functions"
                msg += " " + str(valid_funcs_arity_2)
                raise Exception(msg)
    return suggested_funcs


class Dataset(object):
    """
    A class used to store the dataset of the symbolic regression problem.

    Parameters
    ----------
    path_to_csv_file: string
       Absolute or relative path to the CSV file for the numerical data. The
       rightmost column of the CSV file should be the dependent variable.
       The CSV file should have a header of column names and should NOT
       have a leftmost index column.

    int_max_params: int
        The maximum number of fitting parameters specified in the symbolic
        regression problem.

    path_to_weights: string 
        An absolute or relative path to the CSV for weights of the data points 
        in the CSV found in `path_to_csv`. If `None`, will assume all data 
        points are equally weighted.

    Returns
    -------
    self
        A pyGOURGS.Dataset object, which houses a variety of attributes 
        including the numerical data, the sympy namespace, the data dict used in 
        evaluating the equation string, etc.
    """

    def __init__(self, 
                 path_to_csv_file, 
                 int_max_params, 
                 path_to_weights):
        (dataframe, header_labels) = self.load_csv_data(path_to_csv_file)
        self._int_max_params = int_max_params
        self._dataframe = dataframe
        self._header_labels = header_labels
        x_data, x_labels = self.get_independent_data()
        y_data, y_label  = self.get_dependent_data()
        if path_to_weights is not None:
            (weights_df, empty_labels) = self.load_csv_data(path_to_weights, 
                                                            header=None)            
            self._data_weights = np.squeeze(weights_df.values)
        else:
            self._data_weights = np.ones((len(y_data)))
        illegal_keyphrases = ['nan', 'zoo']
        for illegal in illegal_keyphrases:
            if illegal in (x_labels + y_label):
                raise Exception("'zoo' & 'nan' illegal in variable names")
        self._x_data = x_data
        self._x_labels = x_labels        
        self._y_data = y_data
        self._y_label = y_label        
        if np.std(self._y_data) == 0:
            raise Exception("The data is invalid. All y values are the same.")
        self._param_names, self._params = create_parameter_list(int_max_params)
        self._data_dict = self.get_data_dict()
        self._num_variables = len(self._x_labels)
        self._num_terminals = self._num_variables + int_max_params
        self._terminals_list = (self._param_names +
                                create_variable_list(path_to_csv_file))        

    def make_sympy_namespace(self):
        sympy_namespace = {}
        for variable_name in self._x_labels:
            sympy_namespace[variable_name] = sympy.Symbol(variable_name)
        for param_name in self._param_names:
            sympy_namespace[param_name] = sympy.Symbol(param_name)
        sympy_namespace['add'] = sympy.Add
        sympy_namespace['sub'] = sympy_Sub
        sympy_namespace['mul'] = sympy.Mul
        sympy_namespace['div'] = sympy_Div
        sympy_namespace['pow'] = sympy.Pow
        sympy_namespace['cos'] = sympy.Function('cos')
        sympy_namespace['sin'] = sympy.Function('sin')
        sympy_namespace['tan'] = sympy.Function('tan')
        sympy_namespace['cosh'] = sympy.Function('cosh')
        sympy_namespace['sinh'] = sympy.Function('sinh')
        sympy_namespace['tanh'] = sympy.Function('tanh')
        sympy_namespace['exp'] = sympy.Function('exp')
        sympy_namespace['log'] = sympy.Function('log')
        return sympy_namespace

    def load_csv_data(self, path_to_csv, header=True):
        if header is True:
            dataframe = pandas.read_csv(path_to_csv)
        else:
            dataframe = pandas.read_csv(path_to_csv, header=header)
        column_labels = dataframe.keys()
        return (dataframe, column_labels)

    def get_independent_data(self):
        '''
            Loads all data in self._dataframe except the rightmost column
        '''
        dataframe = self._dataframe
        header_labels = self._header_labels
        features = dataframe.iloc[:, :-1]
        features = np.array(features)
        labels = header_labels[:-1]
        return (features, labels)

    def get_dependent_data(self):
        '''
            Loads only the rightmost column from self._dataframe
        '''
        dataframe = self._dataframe
        header_labels = self._header_labels
        feature = dataframe.iloc[:, -1]
        feature = np.array(feature)
        label = header_labels[-1]
        return (feature, label)

    def get_data_dict(self):
        '''
            Creates a dictionary object which houses the values in the dataset 
            CSV. The variable names in the CSV become keys in this data_dict 
            dictionary.
        '''
        dataframe = self._dataframe
        data_dict = dict()
        for label in self._header_labels:
            data_dict[label] = np.array(dataframe[label].values).astype(float)
            check_for_nans(data_dict[label])
        return data_dict


class SymbolicRegressionConfig(object):
    """
    An object used to store the configuration of this symbolic regression
    problem.

    Parameters
    ----------

    path_to_csv: string
        An absolute or relative path to the dataset CSV file. Usually, this
        file ends in a '.csv' extension.

    path_to_db: string
        An absolute or relative path to where the code can save an output
        database file. Usually, this file ends in a '.db' extension.

    n_functions: list
       A list with elements from the set ['add','sub','mul','div','pow'].
       Defines the functions of arity two that are permitted in this symbolic
       regression run. Default: ['add','sub','mul','div', 'pow']

    f_functions: list
        A list with elements from the set ['cos','sin','tan','cosh','sinh',
        'tanh','exp','log']. Defines the functions of arity one that are
        permitted in this symbolic regression run.
        Default: []

    max_num_fit_params: int
        This specifies the length of the fitting parameters vector. Randomly
        generated equations can have up to `max_num_fit_params` independent
        fitting parameters. Default: 3

    max_permitted_trees: int
        This specifies the number of permitted unique binary trees, which
        determine the structure of random equations. pyGOURGS will consider
        equations from [0 ... max_permitted_trees] during its search. Increasing
        this value increases the size of the search space. Default: 100

    path_to_weights: string 
        An absolute or relative path to the CSV for weights of the data points 
        in the CSV found in `path_to_csv`. If `None`, will assume all data 
        points are equally weighted.           
    
    simplify_solutions: boolean
        Should the equations be run through the simpify function to simplify 
        them and mitigate duplicates
        
    
    Attributes
    ----------
    
    Most are simply the parameters which were passed in. Notably, there is the 
    dataset object, which is not a mere parameter.
    
    self._dataset
        A pyGOURGS.Dataset object, which houses a variety of attributes 
        including the numerical data, the sympy namespace, the data dict used in 
        evaluating the equation string, etc.
    
    Returns
    -------
    self
        A pyGOURGS.SymbolicRegressionConfig object, with attributes 
        self._path_to_csv, 
        self._path_to_db,
        self._n_functions, 
        self._f_functions, 
        self._max_num_fit_params, 
        self._max_permitted_trees,  
        self._path_to_weights, 
        self._simplify_solutions and 
        self._dataset.
    """

    def __init__(self,
                 path_to_csv,
                 path_to_db,
                 n_functions,
                 f_functions,
                 max_num_fit_params,
                 max_permitted_trees,
                 path_to_weights,
                 simplify_solutions):  
        if path_to_db is None:
            path_to_db = create_db_name(path_to_csv)
        self._n_functions = n_functions
        self._f_functions = f_functions
        self._max_num_fit_params = max_num_fit_params
        self._max_permitted_trees = max_permitted_trees        
        self._path_to_csv = path_to_csv
        self._path_to_db = path_to_db
        self._simplify_solutions = simplify_solutions
        is_csv_valid(path_to_csv, True)
        self._path_to_weights = path_to_weights
        if path_to_weights is not None:
            is_csv_valid(path_to_weights)
        self._dataset = Dataset(path_to_csv, 
                                max_num_fit_params, 
                                path_to_weights)


def evalSymbolicRegression(equation_string, SR_config, 
                           scoretype='mean_absolute_error'):
    """
        Evaluates the proposed solution according to its goodness of fit 
        measure, sum of absolute residuals.
    """
    data_dict = SR_config._dataset.get_data_dict()
    independent_vars_vector, x_label = SR_config._dataset.get_independent_data()
    dependent_var_vector, y_label = SR_config._dataset.get_dependent_data()
    # the order of insertion xlabels, then 'weights' matters.
    variables = []
    for i in range(0, len(x_label)):
        variables.append(independent_vars_vector[:,i])
    variables.append(SR_config._dataset._data_weights)
    variables.append(dependent_var_vector)
    variables = tuple(variables)    
    parameters = create_fitting_parameters(SR_config._max_num_fit_params)
    # format the params variables in such a manner as to be dict accessible
    for i in range(0, SR_config._max_num_fit_params):
        equation_string = equation_string.replace('p'+str(i), 
                                                    "params['p" + str(i) + "']")
    # create the lambda function
    # again, the order of 'params', xlabels, 'weights' matters.
    args = str(tuple(['params'] + 
               SR_config._dataset._x_labels.tolist() + 
               ['weights', 'y_true']))
    args = args.replace("'", "")
    args = args.replace(" ", "")
    code = "def equation {args}: return (({code}) - y_true) * weights".format(args=args, 
                                                           code=equation_string)    
    exec(code)
    # fit the parameters
    params = SR_config._dataset._params
    try:
        result = lmfit.minimize(locals()['equation'], 
                                params,
                                args=variables,
                                method='leastsq', 
                                nan_policy='raise')
        residual = result.residual
        params = result.params
    except ValueError:
        residual = np.inf 
    y_true = dependent_var_vector
    y_pred = y_true - residual
    results_dict = regression_results(y_true, y_pred)
    try:
        score = results_dict[scoretype]
    except KeyError:
        raise Exception("Invalid scoretype")
    if score == 0:
        pdb.set_trace()
    return score, params, results_dict
    
    
def simplify_equation_string(eqn_str, dataset):
    """
    Simplify a pyGOURGS equation string into a more human readable format

    Parameters
    ----------
    eqn_str: string
        pyGOURGS equation string

    dataset: pyGOURGS.Dataset
        The dataset object used to generate the `eqn_str`

    Returns
    -------
    eqn_str: string
        A simpler, more human readable version of `eqn_str`

    Notes
    -------
    Uses sympy to perform simplification. The dataset object specifies the sympy
    namespace.
    """
    dataset._sympy_namespace = dataset.make_sympy_namespace()
    s = sympy.sympify(eqn_str, locals=dataset._sympy_namespace)
    try:
        eqn_str = str(sympy.simplify(s))
    except ValueError:
        pass
    if 'zoo' in eqn_str:  # zoo (complex infinity) in sympy
        return 'np.inf'
    if 'nan' in eqn_str:  # nan in sympy
        return 'np.inf'    
    return eqn_str


def solution_saving_worker(queue, n_items, output_db, halloffame):
    """
        Takes solutions from the queue of evaluated solutions, 
        then saves them to the database.
    """
    checkpoint = int(n_items/100) + 1
    for j in range(0, n_items):
        [score, params, metrics, soln] = queue.get()
        halloffame.insert(soln, score, params, metrics)


def main_random(seed, enum, max_tree_complx, SR_config):
    """
        evaluates a randomly generated solution
    """
    soln = enum.uniform_random_global_search_once(max_tree_complx, seed=seed)
    score, params, metrics_dict = evalSymbolicRegression(soln, SR_config)    
    return score, params, soln


def main_random_queued(seed, enum, max_tree_complx, queue, SR_config):
    """
        evaluates a randomly generated solution
        and puts it in the queue 
        used for multiprocessing
    """
    soln = enum.uniform_random_global_search_once(max_tree_complx, seed=seed)
    if SR_config._simplify_solutions == True:
        soln = simplify_equation_string(soln, SR_config._dataset)
    score, params, metrics_dict = evalSymbolicRegression(soln, SR_config)    
    queue.put([score, params, metrics_dict, soln])


def main(soln, SR_config):
    """
        evaluates a proposed solution
    """
    score, params, metrics_dict = evalSymbolicRegression(soln, SR_config)
    return score, params, metrics_dict, soln


def main_queued(soln, SR_config, queue):
    """
        evaluates a proposed solution
        and puts the solution in the queue
        used for multiprocessing
    """
    score, params, metrics_dict = evalSymbolicRegression(soln, SR_config)    
    queue.put([score, params, metrics_dict, soln])


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
    parser.add_argument("csv_path", help="An absolute filepath of the csv that houses your numerical data. Must have a header row of variable names.")
    parser.add_argument("-weights_path", help="An absolute filepath of the csv that will contain the relative weights of the datapoints.")
    parser.add_argument("-funcs_arity_two", help="a comma separated string listing the functions of arity two you want to be considered. Permitted:add,sub,mul,div,pow", default='add,sub,mul,div,pow')
    parser.add_argument("-funcs_arity_one", help="a comma separated string listing the functions of arity one you want to be considered. Permitted:sin,cos,tan,exp,log,sinh,cosh,tanh")
    parser.add_argument("-num_trees", help="pyGOURGS iterates through all the possible trees using an enumeration scheme. This argument specifies the number of trees to which we restrict our search.", type=int, default=1000)
    parser.add_argument("-num_iters", help="An integer specifying the number of search strategies to be attempted in this run", type=int, default=200)
    parser.add_argument("-max_num_fit_params", help="the maximum number of fitting parameters permitted in the generated models", default=3, type=int)
    parser.add_argument("-freq_print", help="An integer specifying how many strategies should be attempted before printing current job status", type=int, default=10)
    parser.add_argument("-deterministic", help="should algorithm be run in deterministic manner?", type=str2bool, default=False)
    parser.add_argument("-exhaustive", help="should algorithm be run in exhaustive/brute-force mode? This can run forever if you are not careful.", type=str2bool, default=False)
    parser.add_argument("-multiprocessing", help="should algorithm be run in multiprocessing mode?", type=str2bool, default=False)
    parser.add_argument("-simplify_solutions", help="should solutions be simplified? time intensive. (default=True)", type=str2bool, default=True)
    parser.add_argument("output_db", help="An absolute filepath where we save results to a JSON database. Include the filename. Extension is typically '.txt'")
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    arguments = parser.parse_args()   
    csv_path = arguments.csv_path
    weights_path = arguments.weights_path
    maximum_tree_complexity_index = arguments.num_trees
    n_iters = arguments.num_iters
    max_num_fit_params = arguments.max_num_fit_params
    frequency_printing = arguments.freq_print
    deterministic = arguments.deterministic
    exhaustive = arguments.exhaustive
    multiproc = arguments.multiprocessing
    output_db = arguments.output_db
    n_funcs = arguments.funcs_arity_two
    n_funcs = n_funcs.split(',')
    n_funcs = check_validity_suggested_functions(n_funcs, 2)
    f_funcs = arguments.funcs_arity_one
    simplify_solutions = arguments.simplify_solutions
    if f_funcs is None or f_funcs == '':
        f_funcs = []
    else:
        f_funcs = f_funcs.split(',')
        f_funcs = check_validity_suggested_functions(f_funcs, 1)
    SR_config = SymbolicRegressionConfig(csv_path, 
                                        output_db, 
                                        n_funcs, 
                                        f_funcs, 
                                        max_num_fit_params, 
                                        maximum_tree_complexity_index, 
                                        weights_path,
                                        simplify_solutions)
    pset = pg.PrimitiveSet()
    for operator_arity_2 in n_funcs:        
        pset.add_operator(operator_arity_2, 2)
    for operator_arity_1 in f_funcs:        
        pset.add_operator(operator_arity_1, 1)
    for terminal in SR_config._dataset._terminals_list:
        pset.add_variable(terminal)
    enum = pg.Enumerator(pset)
    halloffame = HallOfFame(100, output_db)
    if deterministic == False:
        deterministic = None
    best_score = np.inf
    iter = 0
    manager = mp.Manager()
    queue = manager.Queue()
    if exhaustive == True:
        _, weights = enum.calculate_Q(maximum_tree_complexity_index)
        num_solns = int(np.sum(weights))
        txt = input("The number of equations to be considered is " + 
                    str(num_solns) + ", do you want to proceed?" + 
                    " If yes, press 'c' then 'enter'.")
        if txt != 'c':
            print("You input: " + txt + ", exiting...")
            exit(1)
        if multiproc == True:
            jobs = []
            runner = mp.Process(target=solution_saving_worker, 
                             args=(queue, num_solns, output_db, halloffame))
            runner.start()
            for soln in enum.exhaustive_global_search(
                                                 maximum_tree_complexity_index):
                if SR_config._simplify_solutions == True:
                    soln = simplify_equation_string(soln, SR_config._dataset)
                jobs.append(soln)
                iter = iter + 1
                print("\033[K" + "Progress: " + str(iter/num_solns), end='\r')
            results = parmap.map(main_queued, jobs, SR_config=SR_config, 
                                 queue=queue, pm_pbar=True, pm_chunksize=3)
            runner.join()
        elif multiproc == False:
            for soln in enum.exhaustive_global_search(
                                                 maximum_tree_complexity_index):
                if SR_config._simplify_solutions == True:
                    soln = simplify_equation_string(soln, SR_config._dataset)
                score, params, metrics_dict, soln = main(soln, SR_config)
                halloffame.insert(soln, score, params, metrics_dict)
                iter = iter + 1
                if score < best_score:
                    best_score = score
                if iter % frequency_printing == 0:
                    print("\033[K" + "best score of this run:" + 
                          str(best_score), 'iteration:'+ str(iter), end='\r')
        else:
            raise Exception("Invalid value multiproc must be true/false")
    elif exhaustive == False:
        num_solns = n_iters
        if multiproc == True:
            current_time = int(time.time())
            seeds = np.arange(0, n_iters)
            seeds = seeds*current_time
            seeds = seeds.tolist()
            runner = mp.Process(target=solution_saving_worker, 
                             args=(queue, num_solns, output_db, halloffame))
            runner.start()
            results = parmap.map(main_random_queued, seeds, enum=enum, 
                                 max_tree_complx=maximum_tree_complexity_index, 
                                 queue=queue, SR_config=SR_config, 
                                 pm_pbar=True, pm_chunksize=3)
            runner.join()
        elif multiproc == False:
            for soln in enum.uniform_random_global_search(
                                                  maximum_tree_complexity_index, 
                                                   n_iters, seed=deterministic):
                if SR_config._simplify_solutions == True:
                    soln = simplify_equation_string(soln, SR_config._dataset)
                score, params, metrics_dict, soln = main(soln, SR_config)
                if np.isnan(score) == True:
                    score = np.inf
                halloffame.insert(soln, score, params, metrics_dict)
                iter = iter + 1
                if score < best_score:
                    best_score = score
                if iter % frequency_printing == 0:
                    print("\033[K" + "best score of this run:" + 
                          str(best_score), 'iteration:'+ str(iter), end='\r')
        else:
            raise Exception("Invalid multiproc, must be true/false")    
    else:
        raise Exception("Invalid value for exhaustive")
    halloffame.save_to_json()
    halloffame.print_best()
