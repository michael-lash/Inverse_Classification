import numpy as np
from projection_simplex import *

testVec = [1,2,3]
cost = [1,3,6]
l = [0,0,0]
u = [1,1,1]
budget = 2

# use: projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000)
mTV = projection_simplex_bisection(testVec, z=budget, tau=0.0001, max_iter=1000)
print("original vec: "+str(testVec))
print("updated vec: "+str(mTV))
print("budget: "+str(budget))

# use: projection_simplex_bisection_mod(v, z, c, l, u, tau=0.0001, max_iter=1000)
mTV2 = projection_simplex_bisection_mod(np.array(testVec), z=budget, c=np.array(cost), l=np.array(l), u=np.array(u), tau=0.0001, max_iter=1000)
print("mod up vect: "+str(mTV2))
print("cost times mod :"+str(cost*mTV2))
print("sum :"+str(sum(cost*mTV2)))
