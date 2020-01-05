# This code is derived from the DEAP project
# https://github.com/DEAP/deap/blob/master/examples/gp/symbreg.py

import operator
import math
import random
import numpy

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = pg.PrimitiveSet()
pset.add_operator(operator.add, 2)
pset.add_operator(operator.sub, 2)
pset.add_operator(operator.mul, 2)
pset.add_operator(protectedDiv, 2)
pset.add_operator(operator.neg, 1)
pset.add_operator(math.cos, 1)
pset.add_operator(math.sin, 1)
pset.add_terminal('x')
enum = pg.Enumerator(pset)

points=[x/10. for x in range(-10,10)]

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

def evalSymbReg(individual, points, pset):
    # Transform the tree expression in a callable function
    func = compile(individual, pset)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),


if __name__ == "__main__":
    args = sys.argv[1:]
    output_db = args[0]
    n_iters = int(args[1])
    max_score = 0
    iter = 0
    for soln in enum.uniform_random_global_search(10000, n_iters):
        iter = iter + 1 
        score = evalSymbReg(soln, points, pset)[0]
        pg.save_result_to_db(output_db, score, soln)
        if score > max_score:
            max_score = score
        if iter % 10 == 0:
            print(score, max_score, iter)
