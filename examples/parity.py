# This code is derived from the DEAP project
# https://github.com/DEAP/deap/blob/454b4f65a9c944ea2c90b38a75d384cddf524220/examples/gp/parity.py

import random
import operator
import numpy
import sys,os
sys.path.append(os.path.join('..', 'pyGOURGS'))
import pyGOURGS as pg

# Initialize Parity problem input and output matrices
PARITY_FANIN_M = 6
PARITY_SIZE_M = 2**PARITY_FANIN_M

inputs = [None] * PARITY_SIZE_M
outputs = [None] * PARITY_SIZE_M

for i in range(PARITY_SIZE_M):
    inputs[i] = [None] * PARITY_FANIN_M
    value = i
    dividor = PARITY_SIZE_M
    parity = 1
    for j in range(PARITY_FANIN_M):
        dividor /= 2
        if value >= dividor:
            inputs[i][j] = 1
            parity = int(not parity)
            value -= dividor
        else:
            inputs[i][j] = 0
    outputs[i] = parity

pset = pg.PrimitiveSet()
pset.add_operator("operator.and_", 2)
pset.add_operator("operator.or_", 2)
pset.add_operator("operator.xor", 2)
pset.add_operator("operator.not_", 1)
for i in range(0,PARITY_FANIN_M):
    pset.add_variable("BOOL"+str(i))
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

def evalParity(individual, pset):
    func = compile(individual,pset)
    return sum(func(*in_) == out for in_, out in zip(inputs, outputs)),

def main():
    max_score = 0
    iter = 0
    for soln in enum.uniform_random_global_search(10000, 20000):
        iter = iter + 1 
        score = evalParity(soln, pset)[0]
        if score > max_score:
            max_score = score
        if iter % 10 == 0:
            print(score, max_score, iter)
        if score == len(outputs):
            pdb.set_trace()
            print("We have reached a perfect solution")


if __name__ == "__main__":
    main()
