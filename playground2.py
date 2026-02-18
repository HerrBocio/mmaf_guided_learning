import numpy as np
from scipy.optimize import newton

# #delta = 0.025
# #delta=1/.025
# #print(1/delta)
# m=1000
# #print(np.log(1/delta))
# #print(1/np.sqrt(m))
# #print(np.log(1/delta)/np.sqrt(m))


# el = np.array([[1,2,7],[3,4,8],[5,6,9]])
# x= np.array([1,5])
# y=np.matmul(x,el[:-1,:]) +el[-1,:]


# #print(el[:-1,:])
# #print(el[-1,:])
# #print(y)


# el = np.array([[1,3,5],[2,4,6],[7,8,9]])
# x= np.array([1,5])
# y=np.matmul(x,el[:-1,:]) +el[-1,:]

# print(el[:-1,:])
# print(el[-1,:])
# print(y)

m_batches = 1000
p=1
A_estimatedM=[3.840956,3.868912]
lambda_ = A_estimatedM
a_val=[]

for i in [0,1]:
  a= np.ceil(np.log(m_batches)/(2*lambda_[i])+p)
  #a_vgl= np.ceil(- np.log(0.025/(2*eps*m_batches))/l + p )
  #jax.debug.print("a meins {}", a)
  #jax.debug.print("a vgl {}", a_vgl)
  a_val.append(int( a) )

print(a_val)

####################################

lambda_ = 0.6975916
N = 3546

a_search=lambda a : lambda_*(a-p) -1/2* np.log(N/a-19) 
#a_search=lambda x : lambda_*h_t*(x-p) + np.log(0.025*x/(2*eps*N_train)) 

#a_val computed

a_val= int(np.ceil(newton(a_search,1))) # 1=a_0, starting point

print('newton',a_val)