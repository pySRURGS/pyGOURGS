#!/usr/bin/env python
'''
pyGOURGS - Global Optimization via Uniform Random Global Search
Sohrab Towfighi (C) 2019
License: GPL 3.0
https://github.com/pySRURGS/pyGOURGS
'''

class PrimitiveSet(object):
    """    
    A class used to store the terminals and operators used in this global 
    optimization problem.    
    
    Returns
    -------
    self
        A pyGORS.PrimitiveSet object, with attributes
        self._terminals, self._operators.

    """
    def __init__(self):
        self._variables = list()
        self._fitting_parameters = list()
        self._operators = dict()
        self._names = list()

    def add_operator(self, func_handle, arity):
        """
        A method that adds a user-specified operator to the list of operators.

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
        A method that adds a user-specified variable to the list of terminals.

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
        A method that adds a fitting parameter to the list of terminals.

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
    arities = []
    for arity in pset._operators.keys():
        arities.append(arity)
    arities = sorted(arities)
    permitted_arities_indices = list(range(1,len(arities)+1)
    if i == 0:
        tree = '.'    
    else:        
        if i in permitted_arities_indices:
            temp_tree = '['
            arity = permitted_arities_indices[i]
            for i in range(0,arity):
                temp_tree += '.,'
            temp_tree = temp_tree[:-1] + ']'  
            tree = temp_tree
        else:
            n_children = arities[i % len(arities)]
            # deinterleave the number into n_children separate numbers 
            # each of which then can be called to give a child
            left_int, right_int = get_left_right_bits(i)
            left = ith_n_ary_tree(left_int)
            right = ith_n_ary_tree(right_int)
            tree = '[' + left + ', ' + right + ']'
    return tree

        