![Binoculars](image/pyGOURGS.svg)

[![Build Status](https://travis-ci.org/pySRURGS/pyGOURGS.svg?branch=master)](https://travis-ci.org/pySRURGS/pyGOURGS)
[![Coverage Status](https://coveralls.io/repos/github/pySRURGS/pyGOURGS/badge.svg?branch=master)](https://coveralls.io/github/pySRURGS/pyGOURGS?branch=master)
[![License: GPL v3](image/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![python versions](image/python-3_6_3_7-blue.svg)](https://www.python.org)

# Global Optimization by Uniform Random Global Search

This software package solves problems whose solutions can be represented as 
n-ary trees. These problems are typically solved using genetic programming. 
For these problems, there is often little to no relationship between the data
structure representation of a candidate solution and the ultimate performance of 
the candidate solution, once the data structure representation has been 
evaluated to its human readable form. This makes pure random search an 
attractive algorithm with which to solve these kinds of problems. This software 
is aimed at engineers, researchers and data scientists working in data analysis 
and computational optimization.

## Features 

1. Developed and tested on Python 3.6
2. Can be run in deterministic mode for reproducibility
3. Can also run an exhaustive/brute-force search
4. Class instance level memoization of methods for speed
5. API is similar to that of the popular DEAP genetic programming software
6. Example script for the Artificial Ant problem, Multiplexer, and Even Parity problems.

## Getting Started

The software is run using python 3.6. It is run using the terminal.


## Installing

You can install directly from github via the repository.

```
git clone https://github.com/pySRURGS/pyGOURGS.git
cd pyGOURGS
pip install -r requirements.txt --user
```

### An Example: The Artificial Ant Problem

The artificial ant problem is one in which we identify a search strategy for an ant searching 
for breadcrumbs to eat. The crumbs are distributed in a path within a 32 x 32 grid. 
Included in our `/examples/` folder, there are three maps, the `johnmuir_trail.txt`, 
`losaltoshills_trail.txt`, and the `santafe_trail.txt`. By default, the example in `examples/ant.py` 
runs against the `johnmuir_trail.txt`. In the `johnmuir` grid 
shown below, `S` denotes the ant's starting position, `#` denotes a piece of bread, and `.` 
denotes a space without food.

[![John Muir Trail](image/johnmuir.svg)](image/johnmuir.svg)

The ant takes steps according to the search strategy. The ant is 
permitted three base operations, 

1. MOVE forward
2. turn LEFT and 
3. turn RIGHT

The search strategy has functions which define the order in which these base operations are 
executed. These functions are PROGN2, PROGN3, and IF_FOOD_AHEAD. 
- The PROGN2 function takes two arguments and performs them in order. 
- The PROGN3 function similarly takes three arguments and performs them in order. 
- The IF_FOOD_AHEAD function takes two arguments, performing the first if food is ahead and the latter if food is not ahead of the ant. 

Each base operation takes one unit of time to perform. In the included example, 
the simulation stops running after 600 time units.

In the `examples/ant.py` file, we run a search for the ideal search strategy 
using uniform random global search. For the following sections, we refer to code
from the `examples/ant.py` file. 


We begin by instantiating an AntSimulator, each simulation of which we will let 
run for 600 time steps.
```    
ant = AntSimulator(600)
```

We then define the primitives to be used in this problem. The primitives are 
described in terms of n-ary trees. Primitives that are housed in the terminal 
nodes of the tree are dubbed `terminals` (or variables) and primitives that are 
housed in non-terminal nodes are `operators`. pyGOURGS needs to know the number of 
arguments each operator takes, this value is known as the `arity`. This is the 
second argument supplied to `add_operator`.

```
pset = pg.PrimitiveSet()
pset.add_operator("ant.if_food_ahead", 2)
pset.add_operator("prog2", 2)
pset.add_operator("prog3", 3)
pset.add_variable("ant.move_forward()")
pset.add_variable("ant.turn_left()")
pset.add_variable("ant.turn_right()")
```

In the context of the artificial ant problem, as described by Koza (1992), `MOVE`,
`LEFT` and `RIGHT` were terminals, and not operators. In our programming setup, 
these actions are coded as functions with zero arguments. In keeping with the 
original problem specification and since pyGOURGS is not designed to handle 
operators of zero arguments, we simply take these functions and treat them as 
terminals by including the '()' when defining them.

We then instantiate a `pyGOURGS.Enumerator` using the primitive set. The 
enumerator uses the primitives we have defined as a basis for its tree 
enumeration.
```
enum = pg.Enumerator(pset)
```

Every problem solved using pyGOURGS needs to have a custom defined evaluation 
function. pyGOURGS will create potential solutions, but they will be stored as 
strings, which need to be evaluated. For reference, the evaluation function for 
the artificial ant problem is shown below. We create a lambda function using 
the pyGOURGS generated solution

```
def evalArtificialAnt(search_strategy_string):
    # Transform the tree expression to Python code
    routine = eval('lambda : ' + search_strategy_string)
    # Run the generated routine
    ant.run(routine)
    return ant.eaten
```    


Users who wish to try out the completed script can run the bash script and refer 
to the help.

```
$ winpty python ant.py -h
usage: ant.py [-h] [-num_trees NUM_TREES] [-num_iters NUM_ITERS]
              [-freq_print FREQ_PRINT]
              output_db

positional arguments:
  output_db             An absolute filepath where we save results to a SQLite
                        database. Include the filename. Extension is typically
                        '.db'

optional arguments:
  -h, --help            show this help message and exit
  -num_trees NUM_TREES  pyGOURGS iterates through all the possible trees using
                        an enumeration scheme. This argument specifies the
                        number of trees to which we restrict our search.
                        (default: 10000)
  -num_iters NUM_ITERS  An integer specifying the number of search strategies
                        to be attempted in this run (default: 1000)
  -freq_print FREQ_PRINT
                        An integer specifying how many strategies should be
                        attempted before printing current job status (default:
                        10)

```


## API

[Documentation](https://pysrurgs.github.io/pyGOURGS/)

## Author

**Sohrab Towfighi**

## License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE.txt) file for details

## How to Cite

If you use this software in your research, then please cite our papers.

## Community

If you would like to contribute to the project or you need help, then please create an issue.

With regards to community suggested changes, I would comment as to whether it would be within the scope of the project to include the suggested changes. If both parties are in agreement, whomever is interested in developing the changes can make a pull request, or I will implement the suggested changes.

## Acknowledgments

* The example scripts are derived from the DEAP project: [link](https://github.com/DEAP/deap)
* Luther Tychonievich created the algorithm mapping integers to full binary trees: [link](https://www.cs.virginia.edu/luther/blog/posts/434.html), [web archived link](http://web.archive.org/web/20190908010319/https://www.cs.virginia.edu/luther/blog/posts/434.html).
* The icon is derived from the GNOME project and the respective artists. Taken from [link](https://commons.wikimedia.org/wiki/File:Gnome-system-run.svg), [web archived link](https://web.archive.org/web/20161010072611/https://commons.wikimedia.org/wiki/File:Gnome-system-run.svg). License: LGPL version 3.0. 

## References

- Koza JR, Koza JR. Genetic programming: on the programming of computers by means of natural selection. MIT press; 1992.
- Towfighi S. Symbolic regression by uniform random global search. SN Applied Sciences. 2020 Jan 1;2(1):34.