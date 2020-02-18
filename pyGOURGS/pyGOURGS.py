#!/usr/bin/env python
'''
pyGOURGS - Global Optimization by Uniform Random Global Search;
Sohrab Towfighi (C) 2020;
Licence: GPL 3.0;
https://github.com/pySRURGS/pyGOURGS
'''

from functools import lru_cache
from methodtools import lru_cache as mt_lru_cache
import weakref
import mpmath
import numpy as np
import random
import pdb
import sys
from sqlitedict import SqliteDict
import tabulate

class InvalidOperatorIndex(Exception):
    pass

class InvalidTerminalIndex(Exception):
    pass

class InvalidTreeIndex(Exception):
    pass

def count_nodes_in_tree(tree):
    '''
        Given an n-ary tree in string format, counts the number of nodes in the 
        tree               
    '''
    n_terminals = tree.count('..')
    n_operators = tree.count('[')
    n_nodes = n_terminals + n_operators
    return n_nodes

def print_grid(list_iter_0, list_iter_1, funchandle):
    '''
        Function used for debugging purposes during development.
    '''
    grid = np.zeros((len(list_iter_0), len(list_iter_1)))
    for i in range(0,len(list_iter_0)):
        for j in range(0,len(list_iter_1)):
            try: 
                grid[i][j] = funchandle(list_iter_0[i], list_iter_1[j])
            except Exception as e:
                grid[i][j] = -1
    print(grid)

def get_element_of_cartesian_product(*args, repeat=1, index=0):
    """
    Access a specific element of a cartesian product, without needing to iterate
    through the entire product.

    Parameters
    ----------
    args: iterable
        A set of iterables whose cartesian product is being accessed

    repeat: int
        If `args` is only one object, `repeat` specifies the number of times
        to take the cartesian product with itself.

    index: int
        The index of the cartesian product which we want to access

    Returns
    -------
    ith_item: the `index`th element of the cartesian product
    """
    pools = [tuple(pool) for pool in args] * repeat
    if len(pools) == 0:
        return []
    len_product = len(pools[0])
    len_pools = len(pools)
    for j in range(1, len_pools):
        len_product = len_product * len(pools[j])
    if index >= len_product:
        raise Exception("index + 1 is bigger than the length of the product")
    index_list = []
    for j in range(0, len_pools):
        ith_pool_index = index
        denom = 1
        for k in range(j + 1, len_pools):
            denom = denom * len(pools[k])
        ith_pool_index = ith_pool_index // denom
        if j != 0:
            ith_pool_index = ith_pool_index % len(pools[j])
        index_list.append(ith_pool_index)
    ith_item = []
    for index in range(0, len_pools):
        ith_item.append(pools[index][index_list[index]])
    return ith_item

def get_arity_of_term(start_index, tree):
    """
    Returns the arity of the operator which can be placed at index 
    `start_index` within the tree

    Parameters
    ----------
    start_index: int
        Index at which the operator is set to begin. Needs to match to a square
        bracket within the `pyGOURGS` generated string tree

    tree: str
        The solution tree as generated by `pyGOURGS`


    Returns
    -------
    arity: the arity of the operator at `start_index` in `tree`
    """
    bracket_counter = 0
    arity = 1
    if tree[start_index] != '[':
        raise Exception("Start index must point to a square bracket")
    len_solution = len(tree)
    for i in range(start_index, len_solution):
        if tree[i] == '[':
            bracket_counter = bracket_counter + 1
        elif tree[i] == ']':
            bracket_counter = bracket_counter - 1
        if tree[i] == ',' and bracket_counter == 1:
            arity = arity + 1
        if bracket_counter == 0:
            break
    return arity

def mempower(a, b):
    """
    Same as pow, but able to handle extremely large values.

    Parameters
    ----------
    a: int
    b: int

    Returns
    -------
    result: mpmath.ctx_mp_python.mpf (int)
        `a ** b`
    """
    result = int(mpmath.power(a, b))
    return result

class PrimitiveSet(object):
    """    
    A class used to store the terminals and operators used in this global 
    optimization problem.    
    
    Returns
    -------
    self
        A pyGOURGS.PrimitiveSet object

    Example
    -------
    >>>> import pyGOURGS
    >>>> from operator import add, sub, truediv, mul
    >>>> pset.add_operator(add, 2)
    >>>> pset.add_operator(sub, 2)
    >>>> pset.add_operator(truediv, 2)
    >>>> pset.add_variable(1)
    >>>> pset.add_variable(0)    
    """
    def __init__(self):
        self._variables = list()
        self._fitting_parameters = list()
        self._operators = dict()
        self._names = list()

    def add_operator(self, func_handle, arity):
        """
        A method that adds a user-specified operator to the list of operators
        stored in self._operators.

        Parameters
        ----------
        func_handle : str
            The name of a function which will be used in the list of operators.

        arity : integer
            The number of inputs of the function `func_handle`       
        
        Returns
        -------
        None
        """
        if type(arity) != int:
            raise Exception("arity must be int")
        if arity < 1:
            raise Exception("Invalid arity. Must be >= 1.")
        try:
            self._operators[arity]
            self._operators[arity].append(func_handle)
        except KeyError:
            self._operators[arity] = [func_handle]

    def add_variable(self, variable):
        """
        A method that adds a user-specified variable to the list of terminals
        stored in self._variables.

        Parameters
        ----------
        variable: str
            The variable or value which will be used as a terminal. Its type 
            can be anything, but the operators will need to be able to take 
            `variable` as an input. Within pyGOURGS, it is treated as a string,
            but will eventually be evaluated to whatever results from 
            `eval(variable)`.
            
        Returns
        -------
        None
        """
        self._variables.append(variable)
       
    def add_fitting_parameter(self, param_name):
        """
        A method that adds a fitting parameter to the list of terminals
        stored in self._fitting_parameters.

        Parameters
        ----------
        param_name : string 
            The name of the fitting parameter which acts as a terminal
            
        Returns
        -------
        None
        """
        self._fitting_parameters.append(variable)

    def get_terminals(self):
        """
        A method that returns the fitting parameters and variables as one list.

        Parameters
        ----------
        None
            
        Returns
        -------
        terminals : list        
        """
        terminals = self._fitting_parameters + self._variables
        return terminals

    def get_arities(self):
        """
        A method that returns the arities permissible in this search.

        Parameters
        ----------
        None
            
        Returns
        -------
        arities : a sorted list of integers
        """
        return sorted(self._operators.keys())
    
    def get_operators(self):
        """
        A method that returns the operators permissible in this search.

        Parameters
        ----------
        None
            
        Returns
        -------
        operators : list of lists, with elements sorted according to increasing 
            arity
        """
        operators = []
        keys = self._operators.keys()
        sorted_keys = sorted(keys)
        for i in sorted_keys:
            operators.append(self._operators[i])
        return operators

def decimal_to_base_m(v, m):
    """
    A function that converts a decimal number to arbitary base number 

    Parameters
    ----------
    v: int
        The integer in decimal to convert
        
    m: int
        The base of the number system to which we are converting 
        
    Returns
    -------
    result: list of int         
    """
    if v < 0:
        raise Exception("Do not supply negative values")
    def numberToBase(n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]
    if v == 0:
        return [0]
    if m == 1:
        result = []
        for i in range(0, v):
            result.append(1)
    elif m >= 2:
        result = numberToBase(v, m)
    else:
        raise Exception("Invalid m")
    return result

def base_m_to_decimal(v, m):
    """
    A function that converts a base m number to decimal base number 

    Parameters
    ----------
    v: int (or list of int when the output of 'decimal_to_base_m' is considered)
        The integer in base m to convert
        
    m: int
        The base of the number system from which we are converting 
        
    Returns
    -------
    result: int        
    """
    if m > 10 and type(v) is int:
        msg = "Cannot handle m > 10 and type(v) is int. Input v as list"
        raise Exception(msg)
    if m == 1:
        if type(v) is int:
            v = str(v)
        elif type(v) is list:
            v = [str(i) for i in v]
            v = ''.join(v)
        else:
            raise Exception("Invalid type of v")
        result = 0
        for i in v:
            result = result + int(i)            
    elif m >= 2:
        if type(v) is int:
            number = [int(i) for i in str(v)]            
        elif type(v) is list:
            number = v
        else:
            raise Exception("Invalid type of v")
        result = 0
        reversed_number = list(reversed(number))
        for i in range(0,len(number)):
            result = result + reversed_number[i] * m**i            
    else:
        raise Exception("Invalid m")
    return result    

def deinterleave(num, m):
    """
    Given a number `num`, returns the number, deinterleaved, into m folds
    Eg: if `m` were 2, we would be returning the odd and even bits of the number

    Parameters
    ----------
    num : list of ints 
        The number being deinterleaved

    m : int
        An integer denoting the number of folds into which we deinterleave `num`

    Returns
    -------
    m_elements : list of integers

    """
    m_elements = []
    for i in range(0,m):
        m_elements.append([])
    while len(num) % m != 0:
        num.insert(0,0)
    for i in range(0, len(num), m):
        for j in range(0, m):
            m_elements[j].append(str(num[i+j]))
    for j in range(0, m):
        m_elements[j] = int("".join(m_elements[j]))
    return m_elements

class Enumerator(object):
    
    def __init__(self, pset):
        self._pset = pset
        self.assign_variables_from_pset()
        self._results_for_calculate_Q = {}

    def assign_variables_from_pset(self):
        '''
        Helper function used when initiating an Enumerator instance.
        '''
        self._terminals = self._pset.get_terminals()
        self._operators = self._pset._operators
        self._arities = self._pset.get_arities()

    @mt_lru_cache(maxsize=1248)
    def ith_n_ary_tree(self, i):
        """
        Generates the `i`th n-ary tree.
        
        Maps from `i` to the `i`th n-ary tree using an enumeration of possible 
        trees based on the arity of the operators in `pset`.

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        Returns
        -------
        tree: string
            The n-ary tree as a string where `.` denotes terminal, and [ ] 
            define an operator.
        """        
        arities = self._pset.get_arities()
        k = len(arities)
        if i == 0:
            tree = '..'
        elif i in range(1, k+1):
            tree = '['
            m = arities[i-1]
            for i in range(0, m):
                tree += '..,'
            tree = tree[:-1] + ']'
        else:
            e, j = divmod(i-1, k)            
            m = arities[j]            
            e_base_arity = decimal_to_base_m(e, m)
            list_bits = deinterleave(e_base_arity, m)
            list_bits_deci = [base_m_to_decimal(u, m) \
                                 for u in list_bits]
            subtrees = [self.ith_n_ary_tree(x) for x in list_bits_deci]
            tree = '[' + ','.join(subtrees) + ']'
        return tree
    
    @mt_lru_cache(maxsize=1248)
    def calculate_l_i_b(self, i, b):
        """
        Calculates the number of nonterminal nodes, with arity `arities[b]` in 
        tree `i`, called l_i_b

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        b: int 
            Maps via `arities`[b] to the arity of operators being considered

        Returns
        -------
        l_i_b: int            
        """
        arities = self._pset.get_arities()
        k = len(arities)
        if i == 0:
            l_i_b = 0
        elif i in range(1, k+1):
            if b == i-1:
                l_i_b = 1
            else:
                l_i_b = 0
        else:        
            l_i_b = 0
            e, j = divmod(i-1, k) 
            m = arities[j]
            if m == arities[b]:
                l_i_b = l_i_b + 1
            e_base_arity = decimal_to_base_m(e, m)
            list_bits = deinterleave(e_base_arity, m)
            list_bits_deci = [base_m_to_decimal(u, m) \
                                 for u in list_bits]
            for i_deinterleaved in list_bits_deci:
                l_i_b = l_i_b + self.calculate_l_i_b(i_deinterleaved, b)
        return l_i_b

    @mt_lru_cache(maxsize=1248)
    def calculate_G_i_b(self, i, b):
        """
        Calculates the number of possible configurations of operators of arity 
        arities[`b`] in the `i`th tree.

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        b: int
            Maps via `arities[b]` to the arity of operators being considered

        Returns
        -------
        G_i_b : int
            the number of possible configurations of operators of arity `arities`[b]
        """
        arities = self._pset.get_arities()
        f_b = len(self._operators[arities[b]])
        l_i_b = self.calculate_l_i_b(i, b)
        G_i_b = mempower(f_b, l_i_b)
        return G_i_b

    @mt_lru_cache(maxsize=1248)
    def calculate_all_G_i_b(self, i):
        """
        Calculates the number of possible configurations of operators of arity 
        arities[`b`] in the `i`th tree for all values of `b`

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        Returns
        -------
        list_G_i: list
            A list containing the number of possible configurations of operators 
            of arity `arities`[b] 
        """
        arities = self._pset.get_arities()
        k = len(arities)
        list_G_i_b = list()
        for b in range(0, k):
            list_G_i_b.append(self.calculate_G_i_b(i, b))
        return list_G_i_b

    @mt_lru_cache(maxsize=1248)
    def calculate_R_i(self, i):
        """
        Calculates the number of possible configurations of operators in the 
        `i`th tree.

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        Returns
        -------
        R_i: int
            The number of possible configurations of operators in the `i`th 
            tree
        """
        if i == 0:
            return 1
        R_i = 1.0
        all_G_i_b = self.calculate_all_G_i_b(i)
        for G_i_b in all_G_i_b:            
            if G_i_b != 0:
                R_i = R_i * G_i_b
        return R_i

    @mt_lru_cache(maxsize=1248)
    def calculate_a_i(self, i):
        """
        Calculates the number of terminals in the `i`th tree

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        Returns
        -------
        a_i: int
            The number of terminals in the `i`th tree 
        """
        arities = self._pset.get_arities()
        k = len(arities)
        a_i = 0
        if i == 0:
            a_i = 1
        elif i in range(1, k+1):
            a_i = arities[i-1]
        else:
            e, j = divmod(i-1, k) 
            m = arities[j]
            e_base_arity = decimal_to_base_m(e, m)
            list_bits = deinterleave(e_base_arity, m)
            list_bits_deci = [base_m_to_decimal(u, m) \
                                 for u in list_bits]
            for i_deinterleaved in list_bits_deci:
                a_i = a_i + self.calculate_a_i(i_deinterleaved)                
        return a_i
        
    @mt_lru_cache(maxsize=1248)
    def calculate_S_i(self, i):
        """
        Calculates the number of possible configurations of terminals in the 
        `i`th tree.

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        Returns
        -------
        S_i: int
            The number of possible configurations of terminals in the `i`th 
            tree
        """
        m = len(self._pset.get_terminals())
        j_i = self.calculate_a_i(i)
        S_i = mempower(m, j_i)
        return S_i

    @mt_lru_cache(maxsize=1248)
    def calculate_Q(self, N):
        """
        Calculates the number of total number of solutions in the solution space

        Parameters
        ----------
        N: int 
            User specified maximum complexity index

        Returns
        -------
        Q: int
            The number of possible solutions in the solution space
        """ 
        try:
            (Q, weights) = self._results_for_calculate_Q[N]            
        except KeyError:
            pass
        Q = 0
        weights = list()
        for i in range(0, N):
            R_i = self.calculate_R_i(i)
            S_i = self.calculate_S_i(i)
            product = S_i * R_i
            weights.append(product)
            Q = Q + product
        weights = np.array(weights)
        self._results_for_calculate_Q[N] = (Q, weights)
        return Q, weights

    def generate_specified_solution(self, i, r, s, N):        
        """
        Generates a candidate solution given all the indices that map to the
        solution space

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        r: int
            A non-negative integer which will be used to map to a configuration 
            of the operators 

        s: int
            A non-negative integer which will be used to map to a configuration 
            of the terminals       

        N: int 
            User specified maximum complexity index
            
        Returns
        -------
        solution: int
            The candidate solution generated from the supplied indices
        """
        terminals = self._pset.get_terminals()    
        pset = self._pset
        R_i = self.calculate_R_i(i)
        S_i = self.calculate_S_i(i)
        if r >= R_i or r < 0:
            raise InvalidOperatorIndex()
        if s >= S_i or s < 0:
            raise InvalidTerminalIndex()
        if i > N:
            raise InvalidTreeIndex()
        # generate the tree 
        tree = self.ith_n_ary_tree(i)
        # generate the operator configuration 
        G_i_b_values = self.calculate_all_G_i_b(i)
        G_i_b_values = [int(x) for x in G_i_b_values]
        operator_config_indices = np.unravel_index(r, G_i_b_values)
        operator_config = []
        arities = self._pset.get_arities()
        for b in range(0,len(operator_config_indices)):
            z = operator_config_indices[b]            
            arity = arities[b]
            l_i_b = self.calculate_l_i_b(i, b)
            config = get_element_of_cartesian_product(pset._operators[arity],
                                                           repeat=l_i_b, index=z)
            operator_config.append(config)
        # generate the terminal configuration
        a_i = self.calculate_a_i(i)
        terminal_config = get_element_of_cartesian_product(pset.get_terminals(),
                                                           repeat=a_i, index=s)
        # swap in the operators 
        working_tree = tree
        num_opers = working_tree.count('[')
        for i in range(0, num_opers):
            start_index = working_tree.index('[')
            arity = get_arity_of_term(start_index, working_tree)            
            index = arities.index(arity)
            operator = operator_config[index].pop()            
            working_tree = (working_tree[0:start_index] + operator + 
                            '(' + working_tree[start_index+1:])
        working_tree = working_tree.replace(']',')',num_opers)        
        # swap in the terminals 
        num_terminals = working_tree.count('..')
        for i in range(0, num_terminals):
            terminal = str(terminal_config.pop())
            working_tree = working_tree.replace('..', terminal, 1)        
        tree = working_tree 
        return tree

    def uniform_random_global_search_once(self, N, seed=None):
        """
        Generates a random candidate solution

        Parameters
        ----------

        N: int 
            User specified maximum complexity index
            
        Returns
        -------
        candidate_solution: string
            The candidate solution generated from the randomly generated indices
        """
        if seed is not None:
            random.seed(seed)
        terminals = self._pset.get_terminals()        
        pset = self._pset    
        _, weights = self.calculate_Q(N)
        i = random.choices(range(0,N), weights=weights)[0]
        R_i = self.calculate_R_i(i)
        S_i = self.calculate_S_i(i)
        r = random.randint(0, R_i-1)
        s = random.randint(0, S_i-1)
        candidate_solution = self.generate_specified_solution(i, r, s, N)
        return candidate_solution
        
    def uniform_random_global_search(self, N, num_soln, seed=None):
        """
        Yields (this is a generator) a random candidate solutions `num_soln` 
        times.

        Parameters
        ----------

        N: int 
            User specified maximum complexity index

        num_soln: int
            The number of solutions to generate
        
        seed: int
            This value initializes the random number generator. Runs with the 
            same seed will result in identical outputs, so this seed permits us 
            to run the algorithm deterministically.
            
        Yields
        -------
        candidate_solution: string
            The candidate solution generated
        """
        if seed is not None:
            random.seed(seed)
        for j in range(0, num_soln):
            yield self.uniform_random_global_search_once(N)
        
    def exhaustive_global_search(self, N, max_iters=None):
        """
        Yields (this is a generator) candidate solutions incrementally 
        increasing the operator/terminals configurations indices and tree index 
        iterator
        
        Parameters
        ----------

        N: int 
            User specified maximum complexity index
        
        max_iters: int
            The maximum number of solutions which can be considered. Will 
            overrule `N` in terms of cutting off the run.
            
        Yields
        -------
        candidate_solution: string
            The candidate solution generated
        """        
        iter = 1
        for i in range(0, N):
            R_i = int(self.calculate_R_i(i))
            S_i = int(self.calculate_S_i(i))
            for r in range(0, R_i):
                for s in range(0, S_i):
                    if max_iters is not None:
                        if iter > max_iters:
                            return
                        iter = iter + 1 
                    candidate_solution = self.generate_specified_solution(i,
                                                                    r, s, N)                    
                    yield candidate_solution
        
def compile(expr, pset):
    """
    Compiles the `expr` expression

    Parameters
    ----------

    expr: a string of Python code or any object that when
             converted into string produced a valid Python code
             expression.                 

    pset: Primitive set against which the expression is compiled
        
    Returns
    -------
        a function if the primitive set has 1 or more arguments,
         or return the results produced by evaluating the tree
     
    Note
    ----
    This function needs to be copied into the scope of the script where the 
    problem is defined. 
    """    
    code = str(expr)
    if len(pset._variables) > 0:
        args = ",".join(arg for arg in pset._variables)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code)
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("Tree is too long.", traceback)

def initialize_db(path_to_db):
    '''
        Initializes the SqliteDict database file with an initial null value
        for the 'best_result' key
    '''
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        results_dict['best_result'] = None        
    return
    
def save_result_to_db(path_to_db, result, input):
    '''
        Saves results to the SqliteDict file 
    '''
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        results_dict[input] = result    

def check_in_db(path_to_db, input):
    '''
        Checks whether a key already exists in the database
    '''
    with SqliteDict(path_to_db) as results_dict:
        try:
            results_dict[input]
            return True
        except KeyError:
            return False

class Result(object):
    """
    A class to hold a single candidate solution and its performance
    """
    def __init__(self, input, score):
        self._input = input
        self._score = score
        self._nodes = None

class ResultList(object):
    """
    A class to load all results from a pyGOURGS generated database. 

    Returns
    -------
    self: ResultList
    """
    
    def __init__(self, path_to_db):
        self._results = []
        self._path_to_db = path_to_db
        self.load()
        self.sort()        
        self.print()
        
    def load(self):
        """
        Loads a database of solutions into the ResultList
        """        
        with SqliteDict(self._path_to_db) as results_dict:
            keys = results_dict.keys()        
            for input in keys:
                score = results_dict[input]
                my_result = Result(input, score)
                self._results.append(my_result)    
    
    def sort(self):
        """
        Sorts the results in the result list by decreasing value of mean squared 
        error.
        """
        self._results = sorted(self._results, key=lambda x: x._score, reverse=True)

    def count_nodes(self):
        """
        For each result, count the number of nodes in the tree
        """
        for i in range(0,len(self._results)):
            n_nodes = count_nodes_in_tree(self._results[i]._input)
            self._results[i]._nodes = n_nodes
        
    def print(self, top=5, mode='succinct'):
        """
        Prints the score for the top results in the database. Run `self.sort` prior 
        to executing `self.print`.
        
        Parameters
        ----------
                
        top: int 
            The number of results to display. Will be the best models if 
            `self.sort` has been run prior to printing.
                        
        Returns
        -------
        table_string: string
        """
        table = []
        header = ["Score"]
        for i in range(0, top):
            row = [self._results[i]._input, self._results[i]._score]
            table.append(row)
        table_string = tabulate.tabulate(table, headers=header)
        print(table_string)


    