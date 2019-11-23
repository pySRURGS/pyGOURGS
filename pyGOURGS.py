#!/usr/bin/env python
'''
pyGOURGS - Global Optimization via Uniform Random Global Search
Sohrab Towfighi (C) 2019
License: GPL 3.0
https://github.com/pySRURGS/pyGOURGS
'''
from operator import add, sub, truediv, mul
import numpy as np
import pdb

class PrimitiveSet(object):
    """    
    A class used to store the terminals and operators used in this global 
    optimization problem.    
    
    Returns
    -------
    self
        A pyGOURGS.PrimitiveSet object, with attributes
        self._terminals, self._operators.

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


def deinterleave_num_into_k_elements(num, k):
    """
    Given a binary number `num`, returns the number, deinterleaved, into k folds
    Eg: if `k` were 2, we would be returning the odd and even bits of the number. 

    Parameters
    ----------
    num : int
        A number in binary.

    k : int
        An integer denoting the number of folds into which we deinterleave `num`.

    Returns
    -------
    k_elements : list of integers

    """
    k_elements = []
    for i in range(0,k):
        k_elements.append([])
    num = str(num)
    while len(num) % k != 0:
        num = '0' + num
    for i in range(0, len(num), k):
        for j in range(0, k):
            k_elements[j].append(num[i+j])
    for j in range(0, k):
        k_elements[j] = ''.join(k_elements[j])
    return k_elements
    

def binary(num, pre='', length=16, spacer=0):
    ''' formats a number into binary - https://stackoverflow.com/a/16926270/3549879 '''
    return '{0}{{:{1}>{2}}}'.format(pre, spacer, length).format(bin(num)[2:])
    

def ith_n_ary_tree(i, pset):
    """
    Generates the `i`th n-ary tree

    Parameters
    ----------
    i: int
        A non-negative integer which will be used to map to a unique n-ary trees

    Returns
    -------
    tree: string
        The n-ary tree as a string
    """
    k = len(pset._operators.keys())
    arities = pset._operators.keys()
    arities = sorted(arities)
    if i == 0:
        tree = '.'
    else:        
        if i-1 in range(0,k):
            temp_tree = '['
            arity = arities[i-1]
            for i in range(0,arity):
                temp_tree += '.,'
            temp_tree = temp_tree[:-1] + ']'
            tree = temp_tree
        else:
            j = (i - 1) % (len(arities))
            n_children = arities[j]
            i_as_bits = np.base_repr(i-j-k, k)
            deinterleaved_i = deinterleave_num_into_k_elements(i_as_bits,
                                                               n_children)
            deinterleaved_i_deci = [int(x, k) for x in deinterleaved_i]
            subtrees = [ith_n_ary_tree(x, pset) for x in deinterleaved_i_deci]
            tree = '[' + ','.join(subtrees) + ']'
    return tree

                                     
def generate_solution(tree_index, operators_indices, terminals, pset):
    tree = ith_n_ary_tree(i, pset)
    print('fta')
    
if __name__ == '__main__':
    pset = PrimitiveSet()
    pset.add_operator(add, 2)
    pset.add_operator(sub, 6)
    pset.add_operator(truediv, 3)
    pset.add_variable(1)
    list_of_trees = []
    for i in range(0,12):
        tree = ith_n_ary_tree(i, pset) 
        list_of_trees.append(str(tree))
    list_of_trees = list(set(list_of_trees))
    print(len(list_of_trees))
    
                                     
                                     
                                     