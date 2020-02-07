---
title: 'pyGOURGS - global optimization of n-ary tree representable problems using uniform random global search'
tags:
  - Python
  - Global Optimization
  - Heuristic Optimization
  - Genetic Programming
  - Random Search
authors:
  - name: Sohrab Towfighi
    orcid: 0000-0002-3050-8943
    affiliation: "1"
affiliations:
 - name: Faculty of Medicine, University of Toronto
   index: 1
date: 2 January 2020
bibliography: paper.bib
---

# Summary

Global optimization problems are ubiquitous to engineering and the sciences. 
Many such problems are not amenable to analytical techniques and an examination 
of some potential solutions for these problems often suggests that hill climbing 
algorithms would be unable to navigate the jagged and confusing terrain. Despite 
this, genetic programming is often applied to these problems in the hopes that 
it will be able to identify high quality solutions. We suspect that genetic 
programming would perform no better than random search, in agreement with the 
No Free Lunch Theorems [@wolpert1997no], and we devised this software to allow 
us to perform uniform random global search, also known as pure random search, 
on these problems. The challenge lies in creating a system that enumerates all 
the possible solutions, such that we are then able to randomly select from this 
space of solutions, giving each solution the same probability of being selected.

We use an elegant dense enumeration of full binary trees [@Tychonievich:2013] 
and modify it to allow for enumeration of n-ary trees. The enumeration algorithm we 
use is flexible, modifying its enumeration depending on the arity of the 
functions that the user supplies and the number of variables that the user 
supplies. Though intuitive, uniform random global search is proven to 
converge on the ideal solution as the number of iterations tends to infinity 
[@Solis:2012]. The software comes with three ready examples derived from the 
popular DEAP software [@fortin2012deap]. These include the artificial ant 
problem, the even parity problem, and the multiplexer problem. The software 
is the successor to our earlier work [@towfighi2019pysrurgs], but uses a 
different enumeration algorithm that is much more generalizable whereas our 
previous algorithm was only suitable for symbolic regression problems. 

In the seminal work of [@langdon:1998:antspace], they enumerated the 
solution space using brute force and were able to determine that different types
of random search can require differing amounts of computational effort to 
reach a high quality solution. They found that the random search method commonly 
used to generate the initial population of genetic programming solutions performs 
much worse than does uniform random search. We found one prominent modern paper 
which claimed that genetic programming outperformed random search [@Sipper:2018],
when in fact they were comparing genetic programming to a biased type of random 
search which they then put on further unequal footing. This software has broad 
applicability in the examination of the solution space for global optimization 
problems and in the analysis of benchmark problems, as it permits brute force 
computations in addition to random search. This software will be of use to 
researchers looking to compare the performance of their algorithms with that of
pure random search on a wide variety of global optimization problems.

# Acknowledgements

Luther Tychonievich was the original author of the algorithm which mapped 
non-negative integers to unique binary trees. 

# References
