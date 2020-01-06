import os
import sys
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

N = 10000000
_, weights = enum.calculate_Q(N)
values_of_i = []
number_of_configs_at_i = []
trees = []

for i in range(0,N):
    tree = enum.ith_n_ary_tree(i)
    n_nodes = pg.count_nodes_in_tree(tree)
    if i % 10000 == 0:
        print(i, n_nodes)        
    if n_nodes <= 14:
        values_of_i.append(i)
        number_of_configs_at_i.append(weights[i])
        trees.append(tree)
    else:
        #values_of_i.append(i)
        #number_of_configs_at_i.append(0)
        pass

x = list(values_of_i)
y = number_of_configs_at_i
plt.scatter(x,y,s=3)
#plt.yscale('log')
plt.show()

pdb.set_trace()
print("Hi")