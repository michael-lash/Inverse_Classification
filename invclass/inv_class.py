__author__ = "Michael T. Lash, PhD"
__copyright__ = "Copyright 2019, Michael T. Lash"
__credits__ = [None]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Michael T. Lash"
__email__ = "michael.lash@ku.edu"
__status__ = "Prototype" #"Development", "Production"

import tensorflow as tf
import numpy as np
from proj_simplex import proj_simplex


def inv_class(sess, model, i_model, x, b, c, l, u):

    """
	sess: A tensorflow session
	model: A tensorflow model for evaluating f([x_U,x_I,x_D])
	i_model: A tensorflow model for estimating the I features.
	x: The instance to be inverse classified.
	b: budget
	c: cost vector
	l: lower bounds
	u: upper bounds

    """


    

    #Compute the gradient of the model wrt. x_D
    x_D_grad = tf.gradients(model,self.xD)
    x_D_grad = sess.run(x_D_grad,feed_dict={x_D_grad:x.xD,keep_prob:1.0})#[0][0]

    #Compute the gradient of the model wrt. x_I

    #Computer the gradient of the indirect model wrt. x_D


    None


