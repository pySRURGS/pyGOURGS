# This code is derived from the DEAP project
# https://github.com/DEAP/deap/blob/master/examples/gp/multiplexer.py

import random
import operator
import numpy
import sys,os
sys.path.append(os.path.join('..', 'pyGOURGS'))
import pyGOURGS as pg

# Initialize Multiplexer problem input and output vectors

def if_then_else(condition, out1, out2):
    return out1 if condition else out2

MUX_SELECT_LINES = 3
MUX_IN_LINES = 2 ** MUX_SELECT_LINES
MUX_TOTAL_LINES = MUX_SELECT_LINES + MUX_IN_LINES

# input : [A0 A1 A2 D0 D1 D2 D3 D4 D5 D6 D7] for a 8-3 mux
inputs = [[0] * MUX_TOTAL_LINES for i in range(2 ** MUX_TOTAL_LINES)]
outputs = [None] * (2 ** MUX_TOTAL_LINES)

for i in range(2 ** MUX_TOTAL_LINES):
    value = i
    divisor = 2 ** MUX_TOTAL_LINES
    # Fill the input bits
    for j in range(MUX_TOTAL_LINES):
        divisor /= 2
        if value >= divisor:
            inputs[i][j] = 1
            value -= divisor
    
    # Determine the corresponding output
    indexOutput = MUX_SELECT_LINES
    for j, k in enumerate(inputs[i][:MUX_SELECT_LINES]):
        indexOutput += k * 2**j
    outputs[i] = inputs[i][indexOutput]

pset = pg.PrimitiveSet()
pset.add_operator("operator.and_", 2)
pset.add_operator("operator.or_", 2)
pset.add_operator("operator.not_", 1)
pset.add_operator("if_then_else", 3)
pset.add_variable("A0")
pset.add_variable("A1")
pset.add_variable("A2")
pset.add_variable("D0")
pset.add_variable("D1")
pset.add_variable("D2")
pset.add_variable("D3")
pset.add_variable("D4")
pset.add_variable("D5")
pset.add_variable("D6")
pset.add_variable("D7")
enum = pg.Enumerator(pset)

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

def evalMultiplexer(individual, pset):
    func = compile(individual, pset)
    return sum(func(*in_) == out for in_, out in zip(inputs, outputs)),


if __name__ == "__main__":
    args = sys.argv[1:]
    output_db = args[0]
    n_iters = int(args[1])
    max_score = 0
    iter = 0
    for soln in enum.uniform_random_global_search(10000, n_iters):
        iter = iter + 1 
        score = evalMultiplexer(soln, pset)[0]
        pg.save_result_to_db(output_db, score, soln)
        if score > max_score:
            max_score = score
        if iter % 10 == 0:
            print(score, max_score, iter)
