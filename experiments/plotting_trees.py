import pygraphviz as pgv  
import svgutils.transform as sg
import os
import numpy as np
import sys
sys.path.append(os.path.join('..','pyGOURGS'))
import pyGOURGS as pg
import sqlitedict as SqliteDict
import matplotlib.pyplot as plt
import pdb
from nltk import Tree

pset = pg.PrimitiveSet()
pset.add_operator("IF_FOOD", 2)
pset.add_operator("PROG2", 2)
pset.add_operator("PROG3", 3)
pset.add_variable("MOVE")
pset.add_variable("LEFT")
pset.add_variable("RIGHT")
enum = pg.Enumerator(pset)

N = 10000
_, weights = enum.calculate_Q(N)
values_of_i = []
number_of_configs_at_i = []
trees = []
counter = 0


################## DEAP 

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

MOVE=1
LEFT=1
RIGHT=1

def progn(*args):
    for arg in args:
        arg()

def IF_FOOD(arg1, arg2):
    return True

def PROG2(out1, out2): 
    return partial(progn,out1,out2)

def PROG3(out1, out2, out3):     
    return partial(progn,out1,out2,out3)

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()
psetDEAP = gp.PrimitiveSet("MAIN", 0)
psetDEAP.addPrimitive(IF_FOOD, 2)
psetDEAP.addPrimitive(PROG2, 2)
psetDEAP.addPrimitive(PROG3, 3)
psetDEAP.addTerminal(MOVE, name="MOVE")
psetDEAP.addTerminal(LEFT, name="LEFT")
psetDEAP.addTerminal(RIGHT, name="RIGHT")
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalArtificialAnt(individual):
    # Transform the tree expression to functionnal Python code
    routine = gp.compile(individual, pset)
    # Run the generated routine
    ant.run(routine)
    return ant.eaten,

toolbox.register("evaluate", evalArtificialAnt)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



################## END DEAP 
iter = 0
N_figs = 10
for tree in enum.exhaustive_global_search(N, max_iters=N_figs):        
    treeDEAP = gp.PrimitiveTree.from_string(tree, psetDEAP)
    nodes, edges, labels = gp.graph(treeDEAP)        
    ### Graphviz Section ###
    

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree" + str(iter) + ".svg", prog='dot')
    iter = iter + 1


#create new SVG figure
fig = sg.SVGFigure("2000", "5000")

# load matpotlib-generated figures
dict_figs = {}
plots = []
heights = []
for i in range(0,N_figs):
    dict_figs[i] = sg.fromfile("tree" + str(i) + ".svg")
    plots.append(dict_figs[i].getroot())
    heights.append(int(dict_figs[i].height[:-2]))

new_height = 0
for i in range(1,N_figs):
    new_height = np.sum(heights[0:i])
    plots[i].moveto(0,new_height)

# append plots and labels to figure
fig.append(plots)

# save generated SVG files
fig.save("fig_final.svg")