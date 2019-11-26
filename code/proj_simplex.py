__author__ = "Michael T. Lash, PhD"
__copyright__ = "Copyright 2019, Michael T. Lash"
__credits__ = [None]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Michael T. Lash"
__email__ = "michael.lash@ku.edu"
__status__ = "Production"



"""
Implementation of the simplex projection via bisection search described in:

Lash, Lin, Street, and Robinson. 'A budget-constrained inverse classification framework 
for smooth classifiers'. 2017 IEEE Internation Conference on Data Mining Workshops (ICDMW).
IEEE, 2017.

"""

import numpy as np
import sys

def proj_simplex(v, z, c, l, u, tau=0.0001, max_iter=1000):
	"""
	v: np vector to be projected
	z: budget
	c: cost vector
	l: lower bounds of the elements in v
	u: upper bounds of the elements in v
	tau: tolerance
	max_iter: maximum number of iterations to execute.

	"""
	x = np.zeros((v.size))
	lb = 0.0
	ub = 0.0
	temp = Eval(v, c, l, u, lb)
	if(temp <=z):
		x[np.nonzero(((v-lb*c)>0).astype(int))] = np.maximum(np.minimum(v[np.nonzero(((v-lb*c)>0).astype(int))]-lb*c[np.nonzero(((v-lb*c)>0).astype(int))],u[np.nonzero(((v-lb*c)>0).astype(int))]),l[np.nonzero(((v-lb*c)>0).astype(int))])
		x[np.nonzero(((v-lb*c)<=0).astype(int))] = 0.0
		return x
	ub = np.maximum(ub,np.divide(v,c))
	temp = Eval(v, c, l, u, lb)
	theta = (lb+ub)/2
	temp = Eval(v, c, l, u, theta)
	i = -1
	while(((temp - z > tau) or (temp - z < -tau)) and i < max_iter):
		i+=1
		if(temp-z > 0):
			lb = theta
		if(temp - z < 0):
			ub = theta
		theta = (ub+lb)/2
		temp = Eval(v, c, l, u, theta)
	

	x[np.nonzero(((v-theta*c)>0).astype(int))] = np.maximum(np.minimum(v[np.nonzero(((v-theta*c)>0).astype(int))]-theta[np.nonzero(((v-theta*c)>0).astype(int))]*c[np.nonzero(((v-theta*c)>0).astype(int))],u[np.nonzero(((v-theta*c)>0).astype(int))]),l[np.nonzero(((v-theta*c)>0).astype(int))])
	x[np.nonzero(((v-theta*c)<=0).astype(int))] = 0.0
	return x
    
def Eval(v, c, l, u, theta):
	return sum(np.multiply(c,np.maximum(np.maximum(np.minimum(v-theta*c,u),l),0)))

