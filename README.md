# Rank-order-assignment
Assignment for EN.580.437: Biomedical Data Design

Group members: Kuai Yu, Ruitao Hu

## Description
The goal of the project will be to write a piece of software that matches N patients with K doctors. 
Each patient is allowed to
provide a ranked list of their preference for doctors, however doctors are prohibited from
displaying preferences for patients. Thus the code should takes in the following:

● A list of ranked preferences, 1 list for each patient

● A maximum capacity for each doctor (can initially assume the same capacity - note the
total capacity should exceed the number of patients
And the code should return:

● A list of assignments indicating which doctors are to take care of which patients

## Overview
```
.
├── HungarianAlgorithm.py
├── README.md
├── RankOrderAssignment.ipynb
```
* HungarianAlgorithm.py contains the functions for solving linear assignment problem
* RankOrderAssignment.ipynb is the demo script

### Run
Run RankOrderAssignment.ipynb for a quick demo

## Reference

Eason, E. 2021. Hungarian algorithm Introduction &amp; Python implementation. Medium. https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15 

Kuhn, H.W., 1955. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2, 83–97.. doi:10.1002/nav.3800020109

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. 2020. Array programming with NumPy. Nature 585, 357–362 . 
doi: 10.1038/s41586-020-2649-2. (Publisher link).

