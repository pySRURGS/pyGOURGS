import os
import sys
import seaborn as sns
import numpy as np
sys.path.append(os.path.join('..','pyGOURGS'))
import pyGOURGS as pg
import sqlitedict as SqliteDict
import matplotlib.pyplot as plt
import pdb

pset = pg.PrimitiveSet()
pset.add_operator("ant.if_food_ahead", 2)
pset.add_operator("prog2", 2)
pset.add_operator("prog3", 3)
pset.add_variable("ant.move_forward()")
pset.add_variable("ant.turn_left()")
pset.add_variable("ant.turn_right()")
enum = pg.Enumerator(pset)

n = 10
N = 1000000
_, weights = enum.calculate_Q(N)
values_of_i = []
number_of_configs_at_i = []
trees = []

for i in range(0,N):
    tree = enum.ith_n_ary_tree(i)
    n_nodes = pg.count_nodes_in_tree(tree)
    values_of_i.append(i)
    number_of_configs_at_i.append(n_nodes)
    trees.append(tree)

x = list(values_of_i)
y = number_of_configs_at_i
#plt.scatter(x,y)
x = np.array(x)
y = np.array(y)
plt.figure(figsize=(5,3))
plt.scatter(x, y, color='purple', alpha=0.1)
plt.ylabel("number of nodes in tree")
plt.xlabel("index of enumeration scheme")
plt.tight_layout()
plt.show()
