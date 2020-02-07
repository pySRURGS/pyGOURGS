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

S##########.....................
..........#.....................
..........#.....................
..........#.....................
####......#....#................
...#......#..............#######
...#......#.............#.......
...#......#...#.........#.......
...#......#.#...........#.......
...#......#.............#.......
...########.............#.......
...................#####........
...........#......#.............
..............#...#.............
..................#.............
...............#..#.............
............#.....#.............
..................#.............
...........#....................
........#.......................
..................#.............
..................#.............
.......#..........#.............
.....#............#.............
..................#.............
....#.............#.............
....#...........................
....#...........................
....#..####.######..............
................................
................................
................................

The ant takes steps according to the search strategy. The ant is 
permitted three base operations, 

1. MOVE forward
2. turn LEFT and 
3. turn RIGHT

The search strategy is has functions which define the order in which these base operations are 
permitted. These functions are PROGN2, PROGN3, and IF_FOOD_AHEAD. 
- The PROGN2 function takes two arguments and performs them in order. 
- The PROGN3 function similarly takes three arguments and performs them in order. 
- The IF_FOOD_AHEAD function takes two arguments, performing the first if food is ahead and the latter if food is not ahead of the ant. 

Each base operation takes one unit of time to perform. In the included example, the simulation stops running after 
600 time units.

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
