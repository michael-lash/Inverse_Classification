from __future__ import division
from __future__ import print_function

__author__ = "Michael T. Lash, PhD"
__copyright__ = "Copyright 2019, Michael T. Lash"
__credits__ = [None]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Michael T. Lash"
__email__ = "michael.lash@ku.edu"
__status__ = "Prototype" #"Development", "Production"

from absl import flags,app #Consistent with TF 2.0 API
import sys
import tensorflow as tf
import copy
import numpy as np
from invclass.proj_simplex import proj_simplex
import pickle as pkl
from invclass.utils import load_data
from invclass.inv_utils import inv_gradient, inv_gradient_ind, set_parameters, save_result, set_bounds

seed = 1234
tf.random.set_seed(seed)

tf.compat.v1.disable_eager_execution()


FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', '', 'Path to the data. Required.')
flags.DEFINE_string('data_file', '', 'Name of the file containing the data. Must be \
                   a pickle file. Required.')
flags.DEFINE_string('model_file', '', 'File containing inverse class model. Required.')
flags.DEFINE_string('ind_model_file', '', 'File containing the ind model. Required.')
flags.DEFINE_string('save_file', 'invResult.pkl', 'File name to save results to. Saves to data_path.')
flags.DEFINE_boolean('classification', False, 'Classification or regression. Default: False.')
flags.DEFINE_float('budget_start',1,'Starting budget for inverse classification. Default: 1')
flags.DEFINE_float('budget_end',10, 'Ending budget for inverse classification. Set budget_end =\
                     budget_start if only one budget value is desired. Default: 10')
flags.DEFINE_float('budget_interval',1,'Amount by which to increase the budget values from budget_start\
                   to budget_end. Default: 1')
flags.DEFINE_boolean('pos_only', False, 'Whether to only inverse classify those instances that originally\
                     had a negative outcome. Default: False')
flags.DEFINE_integer('max_iters', 50, 'Maximum number of gradient descent iterations. Default: 100')
flags.DEFINE_float('grad_tol', .0001, 'Gradient descent stopping criteria (\epsilon). Default: .0001')
flags.DEFINE_float('lam', 10, 'Initial gradient multiplier. Default: 10')


tf.compat.v1.enable_eager_execution()

np.set_printoptions(threshold=sys.maxsize)

def inv_class(model, ind_model, x, param_dict):

    """
	model: Model for evaluating f([x_U,x_I,x_D])
	ind_model: Model for estimating the I features.
	x: The instance to be inverse classified.
        param_dict: A dictionary of parameters, created from a function in inv_utils, using the
                    .pkl data_file (this is created from train.py during data processing).

    """
    budgets = param_dict['budgets']
    index_dict = param_dict['inds']
    xU_i = index_dict['xU_ind']
    xI_i = index_dict['xI_ind']
    xD_i = index_dict['xD_ind']
    xD_ii = index_dict['xD_ind_ind']

    xU_xD = np.array([np.hstack([x[xU_i],x[xD_i]])])
    #Initial prediction of indirect
    xI_est = ind_model.predict(xU_xD)[0]
    #Initial prediction using indirect    
    x_init = np.array([np.hstack([x[xU_i],xI_est,x[xD_i]])])    
        
    y_hat_init = model.predict(x_init)[0][1]


    #Define the "previous" budget value for use during optimization
    prev_B = 0

    #NP matrix to store optimized xD values
    xD_opt_mat = np.zeros((len(budgets)+1,len(xD_i)), dtype=np.float32)
    xI_opt_mat = np.zeros((len(budgets)+1,len(xI_i)), dtype=np.float32)
    opt_obj_vect = np.zeros((len(budgets)+1,), dtype=np.float32)
    
    xD_opt_mat[0] = x[xD_i]
    xI_opt_mat[0] = xI_est
    opt_obj_vect[0] = y_hat_init
  

    #Compute initial gradients for setting the bounds of ambig. d
    reg_grad_full = inv_gradient(model,x_init)[0]
    #Compute the gradient of the ind_model
    ind_grad_full = inv_gradient_ind(ind_model,xU_xD,num_loss=len(xI_i))
    #Designate partial gradients
    xD_grad = reg_grad_full[xD_i]
    xI_grad = reg_grad_full[xI_i]
    xD_ind_grad = ind_grad_full[:,xD_ii]

    #df/dxD = df/dxD + df/dxI * dg/dxD
    opt_grad = xD_grad + np.matmul(xI_grad,xD_ind_grad)

    #Set bounds and d, c
    d, c, l, u = set_bounds(x,-1*opt_grad,param_dict)


    #Iterate over the budget values
    bud_iter = 0
    for b in budgets: 
        bud_iter+=1
        #Set iteration parameters
        diff = np.inf #Obj func difference between cur and prev iteration of grad descent
        tot_iters = 0 #Cur num iters of grad descent
        best_obj = [y_hat_init] #Cur objective function value
        if bud_iter == 1: #If this is the first budget...
            prev_xD = copy.deepcopy(x[xD_i]) #Use original xD values
            prev_xI = xI_est #Use original estimated xI values
            prev_B = 0
        else: #Otherwise
            prev_xD = xD_opt_mat[0,] #Use 
            prev_xI = xI_opt_mat[0,] #Use originally estimated xI values
            prev_B = budgets[bud_iter-1]

        opt_xD = copy.deepcopy(prev_xD) #Set optimizing xD to previous xD value
        full_opt_x = np.array([np.hstack([x[xU_i],prev_xI,opt_xD])]) #Create full x vector.
        xU_xD_opt = np.array([np.hstack([x[xU_i],opt_xD])]) #Create x vector for ind model.
        gStep = FLAGS.lam #Set lambda for next budget iteration
        #Set b to the difference between the last b and the current
        #b= b-prev_B
        #Create a vector to hold obj func evaluations
        obj_vect = [opt_obj_vect[0]]

        while tot_iters < FLAGS.max_iters and diff > FLAGS.grad_tol:

        

            #Compute the gradient of the model wrt. x
            reg_grad_full = inv_gradient(model,full_opt_x)[0]
            #Compute the gradient of the ind_model
            ind_grad_full = inv_gradient_ind(ind_model,xU_xD_opt,num_loss=len(xI_i))
            #Designate partial gradients
            xD_grad = reg_grad_full[xD_i]
            xI_grad = reg_grad_full[xI_i]
            xD_ind_grad = ind_grad_full[:,xD_ii]

            #df/dxD = df/dxD + df/dxI * dg/dxD
            opt_grad = xD_grad + np.matmul(xI_grad,xD_ind_grad)  
            #Apply gradient
            temp_opt_xD = opt_xD - 1/gStep*opt_grad
            #Multiply by the direction of change
            temp_opt_xD_diff = np.multiply(d,temp_opt_xD - prev_xD) 
            
            #Project the difference
            proj_xD_diff = proj_simplex(temp_opt_xD_diff,b,c,l,u)
            
            #Convert positive diffs to those reflecting direction of change
            proj_xD_diff = np.multiply(d,proj_xD_diff)

            #Update opt_xD
            opt_xD = prev_xD+proj_xD_diff

            #Re-evaluated xI in terms of new xD 
            xU_xD_opt = np.array([np.hstack([x[xU_i],opt_xD])])
            xI_est = ind_model.predict(xU_xD_opt)[0]
          
            #Re-evaluate obj function using xI_est and xD_opt
            full_opt_x = np.array([np.hstack([x[xU_i],xI_est,opt_xD])])

            cObj = model.predict(full_opt_x)[0][1]
            
            while (cObj > obj_vect[-1]) and gStep < 1000:
                #In case we have haven't exceed the previous iteration
                gStep = gStep * 2
                
                #Compute the gradient of the model wrt. x
                reg_grad_full = inv_gradient(model,full_opt_x)[0]
                #Compute the gradient of the ind_model
                ind_grad_full = inv_gradient_ind(ind_model,xU_xD_opt,num_loss=len(xI_i))
                #Designate partial gradients
                xD_grad = reg_grad_full[xD_i]
                xI_grad = reg_grad_full[xI_i]
                xD_ind_grad = ind_grad_full[:,xD_ii]

                #df/dxD = df/dxD + df/dxI * dg/dxD
                opt_grad = xD_grad + np.matmul(xI_grad,xD_ind_grad)
                #Apply gradient

                temp_opt_xD = opt_xD - 1/gStep*opt_grad
                temp_opt_xD_diff = np.multiply(d,temp_opt_xD - prev_xD) #Multiply directions times difference
                #Project the difference
                proj_xD_diff = proj_simplex(temp_opt_xD_diff,b,c,l,u)
                #Convert positive diffs to those reflecting direction of change
                proj_xD_diff = np.multiply(d,proj_xD_diff)
                #Update opt_xD
                opt_xD = prev_xD+proj_xD_diff

                #Re-evaluated xI in terms of new xD
                xU_xD_opt = np.array([np.hstack([x[xU_i],opt_xD])])

                xI_est = ind_model.predict(xU_xD_opt)[0]

                #Re-evaluate obj function using xI_est and xD_opt
                full_opt_x = np.array([np.hstack([x[xU_i],xI_est,opt_xD])])

                cObj = model.predict(full_opt_x)[0][1]


            obj_vect.append(cObj)
            #Check for objective convergence
            diff = (obj_vect[-2]-obj_vect[-1])/obj_vect[-2]
            #Decrement gradient step
            gStep = gStep/1.5

        #Now set appropriate vars for next budget iteration
        xD_opt_mat[bud_iter] = opt_xD
        xI_opt_mat[bud_iter] = xI_est
        if obj_vect[-1] > obj_vect[-2]:
            opt_obj_vect[bud_iter] = obj_vect[-2]
        else:
            opt_obj_vect[bud_iter] = obj_vect[-1]
    return_dict={"obj":opt_obj_vect,
                 "xD":xD_opt_mat,
                 "xI":xI_opt_mat,
                }
    return return_dict

def main(argv):
    print("Loading data...")
    data_dict = load_data(FLAGS.data_path,FLAGS.data_file,file_type="pkl")
    print("Done loading data. Loading models...")
    reg_model = tf.keras.models.load_model(FLAGS.data_path+FLAGS.model_file)
    ind_model = tf.keras.models.load_model(FLAGS.data_path+FLAGS.ind_model_file)       
    print("Done loading model. Executing inverse classification...")
    inv_data = data_dict['test']
    X_inv = inv_data['X']
    X_ids = inv_data['ids']
    param_dict = set_parameters(data_dict)


    if FLAGS.pos_only:
        inv_inds = np.where(inv_data['target']==1)[0]
    else:
        inv_inds = [i for i in range(X_inv.shape[0])]
    
    #Take gradient of all instances to test for zero gradients
    grads = inv_gradient(reg_model,X_inv[inv_inds])
    nz_grads = np.nonzero(grads)
    nz_inds = list(set(nz_grads[0]))
    print("Total test instances w/ non-zero grads: {}".format(len(nz_inds)),
           "out of {} total instances".format(grads.shape[0]))
    
    result_dict = {"budgets":param_dict['budgets'],'ids':[]}
    improv_mat = np.zeros((len(inv_inds),len(param_dict['budgets'])+1))
    for i,idv in enumerate(inv_inds):
        inv_dat = inv_class(reg_model,ind_model,X_inv[idv],param_dict)
        result_dict['ids'].append(X_ids[idv]) 
        result_dict[X_ids[idv]] = inv_dat
        improv_mat[i,] = inv_dat['obj']
    
    avg_opt = np.mean(improv_mat,axis=0)
    budgets = param_dict['budgets']
    budgets.insert(0,0)
    res_mat = np.vstack([budgets,avg_opt])
    np.set_printoptions(precision=3)
    print("Average probability by budget:\n {}".format(res_mat))
    save_result(result_dict)



if __name__ == '__main__':
    app.run(main)

