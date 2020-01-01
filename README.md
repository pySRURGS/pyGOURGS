![Binoculars](image/pyGOURGS.svg)

### Global Optimization by Uniform Random Global Search

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
6. Example script for the Artificial Ant problem, simple Symbolic Regression, Multiplexer, and Even Parity problems.

## Getting Started

The software is run using python 3.6. It is run using the terminal.


### Installing

Install using pip from the terminal.

```
pip install pyGOURGS
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

* Luther Tychonievich created the algorithm mapping integers to full binary trees: [link](https://www.cs.virginia.edu/luther/blog/posts/434.html), [web archived link](http://web.archive.org/web/20190908010319/https://www.cs.virginia.edu/luther/blog/posts/434.html).
