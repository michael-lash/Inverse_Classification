from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import pandas as pd
import pickle as pkl
import csv
import sys
import tensorflow as tf

from absl import flags,app #Consistent with TF 2.0 API

FLAGS = flags.FLAGS


def make_model(data_dict,hidden_units,indirect_model,categorical=True):

    train_dat = data_dict['train']
    opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    #For training indirect models
    if indirect_model:
        in_dim = len(data_dict['xU_ind'])+len(data_dict['xD_ind'])        
        out_dim = len(data_dict['xI_ind'])
        model = tf.keras.models.Sequential()
        if FLAGS.hidden_units > 0:
            model.add(tf.keras.layers.Dense(FLAGS.hidden_units,input_dim=in_dim,activation='relu'))
            model.add(tf.keras.layers.Dense(out_dim,input_dim=FLAGS.hidden_units,activation='relu'))
        else:
            model.add(tf.keras.layers.Dense(out_dim,input_dim=in_dim,activation='relu'))
        
        model.compile(loss="mse",optimizer=opt, metrics=["mse","mae"])
        return model

    #For training regular models
    in_dim = train_dat['X'].shape[1]
    model = tf.keras.models.Sequential()
    if categorical:
        if FLAGS.hidden_units > 0:
            model.add(tf.keras.layers.Dense(FLAGS.hidden_units,input_dim=in_dim,activation='relu'))
            model.add(tf.keras.layers.Dense(2,input_dim=FLAGS.hidden_units,activation='softmax'))
        else:
            model.add(tf.keras.layers.Dense(2,input_dim=in_dim,activation='softmax'))
        model.compile(loss='binary_crossentropy',optimizer=opt, 
                      metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryAccuracy()])
    else:
        if FLAGS.hidden_units > 0:
            model.add(tf.keras.layers.Dense(FLAGS.hidden_units,input_dim=in_dim,activation='relu'))
            model.add(tf.keras.layers.Dense(1,input_dim=FLAGS.hidden_units,activation='relu'))
        else:
            model.add(tf.keras.layers.Dense(1,input_dim=in_dim,activation='relu'))
        model.compile(loss='mse',optimizer=opt)
    return model




def load_indices(data_path,util_file):
    """
        data_path: Path to data files.

        util_file: Name of the file containing the index designations, cost
                   parameters, and direction of change parameters. Should be
                    of the form:

                        index, designation, cost increase, cost decrease, direction
                      
                        e.g.:

                        0,id,,,
                        1,dir,0,2,-1
                        2,dir,3,0,1
                        3,dir,4,3,0
                        4,unch,,,
                        5,ind,,,
                         ...
                        p,target,,,

    """

    unch_indices = []
    ind_indices = []
    dir_indices = []
    cost_inc = []
    cost_dec = []
    direct_chg = []
    id_ind = -1
    target_ind = -1
    with open(data_path+util_file,'rU') as rF:
        fReader = csv.reader(rF,delimiter=',')
        for i, row in enumerate(fReader):
            if row[1] == 'id':
                id_ind = int(row[0])
            elif row[1] == 'target':
                target_ind = int(row[0])
            elif row[1] == 'ind':
                ind_indices.append(int(row[0]))
            elif row[1] == 'unch':
                unch_indices.append(int(row[0]))
            elif row[1] == 'dir':
                dir_indices.append(int(row[0]))
                cost_inc.append(int(row[2]))
                cost_dec.append(int(row[3]))
                direct_chg.append(int(row[4]))
            else:
                raise Exception("Problem loading index file. Unrecognized designation '{}' found on row\
                          {}".format(row[0],str(i+1)))

    return unch_indices,ind_indices,dir_indices,cost_inc,cost_dec,direct_chg,id_ind,target_ind

def load_data(data_path,data_file,file_type="csv",unchange_indices=[],indirect_indices=[],
                direct_indices=[],id_ind=0,target_ind=-1,seed=1234,val_prop=0.10,test_prop=0.10,
                imbal_classes=False,opt_params={},save_file=""):

    """
        data_path: Path to the data file. The output data will be written to this 
		   location.

        data_file: File containing the data to be loaded.

        file_type: The type of file, either 'csv' or 'pkl'.

                   (1 ) 'csv' assumes the following:

                         a. Has a header and is the first line in the file.
                         b. The first column identifies the instance.
                         c. The last column is the target variable.
                         d. ALL VARIABLES ARE NUMERIC (including identifiers
                            and target).

                   (2) 'pkl' file type is assumed to have been generated
                        according to this code.
        
        unchange_indices: The indices onf the unchangeable features.

        indirect_indices: The indices of the indirectly changeable features.
 
        direct_indices: The indices of the directly changeable features

        seed: Seed to randomly partition data.

        val_prop: Proportion of data to be used for the validation set.

        test_prop: Proportion of data to be used for the test set.

        imbal_classes: Boolean. Whether or not there is class imbalance. If
                       set to True, then we will stratify the positive class
                       (assumed to be the imbalanced class). To ensure that
                       positive samples are present in the train, validation,
                       and test sets.

    """

    if file_type == "pkl":
        with open(data_path+data_file,'rb') as rF:
            load_data = pkl.load(rF)
            return load_data
    
    elif file_type == "csv":
        sep=","
    else:
        raise Exception("Unsupoorted file type {}. Support file types are 'csv' and 'pkl'.".format(file_type))

    dset_df = pd.read_csv(data_path+data_file,sep=sep)

    header = dset_df.columns


    id_col_name = header[id_ind]
    target_col_name = header[target_ind]
    indirect_col_names = header[indirect_indices]
    direct_col_names = header[direct_indices]
    unchange_col_names = header[unchange_indices]

    dset_ids = dset_df[id_col_name].values
    dset_targets = dset_df[target_col_name].values
    X_data = dset_df.drop([id_col_name, target_col_name],axis=1)
    

    unchange_indices = [X_data.columns.get_loc(c) for c in unchange_col_names]
    indirect_indices = [X_data.columns.get_loc(c) for c in indirect_col_names]
    direct_indices = [X_data.columns.get_loc(c) for c in direct_col_names]

    X_data = X_data.values    


    #Randomly define train, val, test indices according to test_prop, val_prop
    np.random.seed(seed=seed)

    if imbal_classes == False:    

        nfull = dset_ind.shape[0]
        test_n = round(nfull*test_prop)
        val_n = round(nfull*val_prop)
        all_indices = [i for i in range(nfull)]
        val_test_indices = np.random.choice(nfull,size=test_n+val_n,replace=False)

        val_indices = [val_test_indices[i] for i in range(val_n)]
        test_indices = [val_test_indices[i] for i in range(val_n,val_n+test_n)]

        train_indices = list(set(all_indices) - set(val_test_indices))


    else:

        pos_inds = np.where(dset_targets == 1)[0]
        neg_inds = np.where(dset_targets == 0)[0]   
  
        npos = pos_inds.shape[0]
        nneg = neg_inds.shape[0]

        test_n_pos = round(npos*test_prop)
        test_n_neg = round(nneg*test_prop)

        val_n_pos = round(npos*val_prop)
        val_n_neg = round(nneg*val_prop)

        val_test_pos_indices = np.random.choice(pos_inds,size=test_n_pos+val_n_pos,replace=False)
        val_test_neg_indices = np.random.choice(neg_inds,size=test_n_neg+val_n_neg,replace=False)

        val_pos_indices = [val_test_pos_indices[i] for i in range(val_n_pos)]
        test_pos_indices = [val_test_pos_indices[i] for i in range(val_n_pos,val_n_pos+test_n_pos)]

        val_neg_indices = [val_test_neg_indices[i] for i in range(val_n_neg)]
        test_neg_indices = [val_test_neg_indices[i] for i in range(val_n_neg,val_n_neg+test_n_neg)]

        train_pos_indices = list(set(pos_inds) - set(val_test_pos_indices))
        train_neg_indices = list(set(neg_inds) - set(val_test_neg_indices))

        val_indices = val_pos_indices + val_neg_indices
        test_indices = test_pos_indices + test_neg_indices
        train_indices = train_pos_indices + train_neg_indices
      

    #Partition data into train,val,test according to the above defined indices
    
    #Train
    train_X = X_data[train_indices]    
    train_target = dset_targets[train_indices]
    train_ids = dset_ids[train_indices]

    #Obtain normalization values
    min_X = np.amin(train_X,axis=0)
    max_X = np.amax(train_X,axis=0) 

    #Normalize training data
    norm_train_X =np.divide(train_X - min_X,max_X - min_X)
    
   
    train_dict = {"X":norm_train_X, "target":train_target, "ids":train_ids}

    #Val
    val_X= X_data[val_indices]
    val_target = dset_targets[val_indices]
    val_ids = dset_ids[val_indices]

    #Normailze validation data
    norm_val_X = np.divide(val_X - min_X,max_X - min_X)

    val_dict = {"X":norm_val_X, "target":val_target, "ids":val_ids}

    #Test
    test_X = X_data[test_indices]
    test_target = dset_targets[test_indices]
    test_ids = dset_ids[test_indices]

    #Normalize test data
    norm_test_X = np.divide(test_X - min_X,max_X - min_X)


    test_dict = {"X":norm_test_X, "target":test_target, "ids":test_ids}


    return_dict = {'train':train_dict,
                   'val':val_dict,
                   'test':test_dict,
                   'train_indices':train_indices,
                   'val_indices':val_indices,
                   'test_indices':test_indices,
                   'min_X':min_X,
                   'max_X':max_X,
                   'opt_params':opt_params,
                   'xU_ind':unchange_indices,
                   'xI_ind':indirect_indices,
                   'xD_ind':direct_indices
                   }

    #If a save file is defined, write the defined data out.
    if save_file != "":
        with open(data_path+save_file,'wb') as sF:
            pkl.dump(return_dict,sF)

    return return_dict


