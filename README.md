# Inverse Classification

Please cite the following papers if you use this code in your research:

1. Michael T. Lash, Qihang Lin, W. Nick Street, and Jennifer G. Robinson. "A budget-constrained inverse classification framework for smooth classifiers." *2017 IEEE International Conference on Data Mining Workshops (ICDMW)*. IEEE, 2017.

2. Michael T. Lash, Qihang Lin, W. Nick Street, Jennifer G. Robinson, and Jeffrey W. Ohlmann. "Generalized inverse classification." *Proceedings of the 2017 SIAM International Conference on Data Mining*. Society for Industrial and Applied Mathematics, 2017.

## Description

This repository provides a suite of several functions that can be used to perform inverse classification on an induced Tensorflow model.

Currently, the code only supports the use of gradient-based optimization via projected gradient descent. The code projects onto a truncated, weighted L_1 ball (i.e., the constraints on the inverse classification process are linear w.r.t. the budget value B).

In the future I will add heurstic optimization and projection onto an L_2 ball (i.e., for constraints that are quadratic).
