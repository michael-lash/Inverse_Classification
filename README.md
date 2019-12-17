# Inverse Classification

Please cite the following papers if you use this code in your research:

1. Michael T. Lash, Qihang Lin, W. Nick Street, and Jennifer G. Robinson. "A budget-constrained inverse classification framework for smooth classifiers." *2017 IEEE International Conference on Data Mining Workshops (ICDMW)*. IEEE, 2017.

2. Michael T. Lash, Qihang Lin, W. Nick Street, Jennifer G. Robinson, and Jeffrey W. Ohlmann. "Generalized inverse classification." *Proceedings of the 2017 SIAM International Conference on Data Mining*. Society for Industrial and Applied Mathematics, 2017.

## Description

This repository provides a suite of several functions that can be used to perform inverse classification on an induced Tensorflow model. There is also code to train models -- both "regular" and "indirect" -- that can be used with provided inverse classification functions.

Currently, the code only supports the use of gradient-based optimization via projected gradient descent. The code projects onto a truncated, weighted L_1 ball (i.e., the constraints on the inverse classification process are linear w.r.t. the budget value B).

In the future I will add heurstic optimization and projection onto an L_2 ball (i.e., for constraints that are quadratic).

## Use

Please examine the files "stud_port_class.csv" and "stud_indices.csv". These are the two files that are needed to train models and then run inverse classification. The provided files are processed "Student Peformance" dataset data, originally found on the machine learning repository at this [link](https://archive.ics.uci.edu/ml/datasets/student+performance). Student Performance was one the datasets used in both the ICDMW and SDM papers.

- stud_port_class.csv: Contains the data.
   
- stud_indices.csv: Designates each feature as being the id, target, or belonging to the unchangeable, indirectly changeable, or directly changeable feature groups. There are also costs +/- imposed and direction of change. Costs and directions must be specified for each directly changeable feature.

**To run the code:**

1. Use train.py to train a "regular" model. Please see the file "example_train.sh" for an example on how to use this code. Also please examine train.py for additional parameters that can be specified. The data will automatically be randomly partitioned into train/validation/test with the test set being used for inverse classification later on. There will be pickle file created that holds all of the necessary data and parameters to be used during the inverse classification process.

2. Use train.py to train an "indirect" model to predict the indirect features. See "example_ind_train.sh" for an example on how to use this code and, again, please also refer to the train.py code for additional parameters that can be specified.

3. Place the models ("regular" and "indirect") you would like to use in the your data directory).

4. Use inv_class.py to execute inverse classification.  Please see "example_inv_class.sh" for an example. Note that the file "./example_data/pos_only-processed_stud-invResult.pkl" contains the results of executing inverse classification. The example_data directory also contains two trained models ("regular" and "indirect").
