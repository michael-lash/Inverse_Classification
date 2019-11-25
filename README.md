# Inverse Classification

## Description

This repository provides a suite of several functions that can be used to perform inverse classification on an induced Tensorflow model.

Currently, the code only supports the use of gradient-based optimization via projected gradient descent. The code projects onto a truncated, weighted L_1 ball (i.e., the constraints on the inverse classification process are linear w.r.t. the budget value B).

In the future I will add heurstic optimization and projection onto an L_2 ball (i.e., for constraints that are quadratic).
