#!/usr/bin/env python
'''
pyGOURGS - Global Optimization via Uniform Random Global Search
Sohrab Towfighi (C) 2019
License: GPL 3.0
https://github.com/pySRURGS/pyGOURGS
'''

import mpmath
import numpy as np
import pdb

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


def mempower(a, b):
    """
    Same as pow, but able to handle extremely large values, and memoized.

    Parameters
    ----------
    a: int
    b: int

    Returns
    -------
    result: mpmath.ctx_mp_python.mpf (int)
        `a ** b`
    """
    result = mpmath.power(a, b)
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
        func_handle : function or builtin_function_or_method
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
        variable
            The variable or value which will be used as a terminal. Its type 
            can be anything, but the operators will need to be able to take 
            `variable` as an input.
            
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
    def numberToBase(n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]
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
    v: int (or list of int when the output of `decimal_to_base_m` is considered)
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
            m_elements[j].append(num[i+j])
    for j in range(0, m):
        m_elements[j] = int("".join(map(str, m_elements[j])))
    return m_elements
    

class Enumerator(object):
    
    def __init__(self, pset):
        self._pset = pset
        self.assign_variables_from_pset()

    def assign_variables_from_pset(self):
        self._terminals = self._pset.get_terminals()
        self._operators = self._pset._operators
        self._arities = self._pset.get_arities()

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

        pset: pyGOURGS.PrimitiveSet object, which specifies the nature of the 
            optimization problem

        Returns
        -------
        tree: string
            The n-ary tree as a string where `.` denotes terminal, and [ ] 
            define an operator.
        """
        k = len(self._pset._operators.keys())
        arities = self._pset.get_arities()
        if i == 0:
            tree = '.'
        elif i in range(1, k+1):
            temp_tree = '['
            arity = arities[i-1]
            for i in range(0, arity):
                temp_tree += '.,'
            temp_tree = temp_tree[:-1] + ']'
            tree = temp_tree
        else:
            j = (i - 1) % k
            m = arities[j]
            v = i-j-k
            v_m = decimal_to_base_m(v, k)
            deinterleaved_v_m = deinterleave(v_m, m)
            deinterleaved_v_d = [base_m_to_decimal(u, k) \
                                 for u in deinterleaved_v_m]
            subtrees = [self.ith_n_ary_tree(x) for x in deinterleaved_v_d]
            tree = '[' + ','.join(subtrees) + ']'
        return tree

    def calculate_l_i_b(self, i, b):
        """
        Calculates the number of operators with arity `b` in tree `i`, called 
        l_i_b

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        b: int 
            Maps via arities[`b`] to the arity of operators being considered

        Returns
        -------
        l_i_b: int
            the number of operators with arity `b` in tree `i`, called l_i_b
        """
        pset = self._pset
        k = len(self._pset._operators.keys())
        arities = pset.get_arities()
        if i == 0:
            l_i_b = 0
            return l_i_b
        if b <= k-1:
            if b == i-1:
                l_i_b = 1
            else:
                l_i_b = 0
            return l_i_b
        else:        
            l_i_b = 0
            j = (i - 1) % (len(arities))
            n_children = arities[j]
            # raise exception if i-j-k < 0:
            if i-j-k < 0:
                raise Exception("Invalid setup")    
            i_as_bits = np.base_repr(i-j-k, k)
            deinterleaved_i = deinterleave_num_into_k_elements(i_as_bits,
                                                               n_children)
            deinterleaved_i_deci = [int(x, k) for x in deinterleaved_i]
            for i_deinteleaved in deinterleaved_i_deci:
                l_i_b = l_i_b + self.calculate_l_i_b(i_deinteleaved, pset)
        return l_i_b

    def calculate_G_i_b(self, i, b):
        """
        Calculates the number of possible configurations of operators of arity 
        `b` in the `i`th tree.

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
            the number of possible configurations of operators of arity
        """
        arities = self._arities
        pdb.set_trace()
        f_b = len(self._operators[arities[b]])
        l_i_b = self.calculate_l_i_b(i, b)
        if l_i_b == 1:
            pdb.set_trace()
        G_i_b = mempower(f_b, l_i_b)
        return G_i_b

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
        k = len(self._pset._operators.keys())
        R_i = mpmath.mpf(1.0)
        for b in range(0, k):
            R_i = R_i * self.calculate_G_i_b(i, b)
        return R_i

    def calculate_j_i(self, i):
        """
        Calculates the number of terminals in the `i`th tree

        Parameters
        ----------
        i: int
            A non-negative integer which will be used to map to a unique n-ary 
            trees

        Returns
        -------
        j_i: int
            The number of terminals in the `i`th tree 
        """
        k = len(self._pset._operators.keys())
        arities = self._pset.get_arities()
        j_i = 0
        if i == 0:
            return 1
        else:        
            if i-1 in range(0,k):
                j_i = arities[i-1]
                return j_i
            else:
                j = (i - 1) % (len(arities))
                n_children = arities[j]
                i_as_bits = np.base_repr(i-j-k, k)
                deinterleaved_i = deinterleave_num_into_k_elements(i_as_bits,
                                                                   n_children)
                deinterleaved_i_deci = [int(x, k) for x in deinterleaved_i]
                for i_deinteleaved in deinterleaved_i_deci:
                    j_i = j_i + self.calculate_j_i(i_deinteleaved)                
        return j_i
        

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
        j_i = self.calculate_j_i(i)
        S_i = mempower(m, j_i)
        return S_i

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
        Q = 0
        for i in range(0, N):
            R_i = self.calculate_R_i(i)
            S_i = self.calculate_S_i(i)
            Q = Q + S_i * R_i
        return Q

if __name__ == '__main__':
    pset = PrimitiveSet()
    pset.add_operator(add, 2)
    pset.add_operator(sub, 2)
    pset.add_operator(truediv, 2)
    pset.add_variable(1)
    enum = Enumerator(pset)
    i_list = range(0,10)
    b_list = range(0,10)
    print_grid(i_list, b_list, enum.calculate_G_i_b)
    