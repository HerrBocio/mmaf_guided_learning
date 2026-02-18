import numpy as np
from scipy.optimize import newton

print("lambda = A*min(2,c)/(2c)")
print("GAU & NIG")

A_estimatedM=[3.840956,3.868912]#,]#,[3.840956]#
c_estimatedM=[1,1]
m_batches=1000
eps=3.
p=1

###

lambda_ = []
a_val =[]

for i in [0,1]:
  l=A_estimatedM[i] * np.minimum(2.0, c_estimatedM[i]) / (2*c_estimatedM[i])
  lambda_.append(l)

  a= np.ceil(- np.log(0.025/(2*eps*m_batches))/l + p )
  a_val.append(int( a) )

print("exp(-lambda(a-p))<= delta/(2*m*eps) -> a_val = ",a_val)

###

lambda_ = []
a_val =[]

for i in [0,1]:
  l=A_estimatedM[i] * np.minimum(2.0, c_estimatedM[i]) / (2*c_estimatedM[i])
  lambda_.append(l)

  a= np.ceil(- np.log(1/(2*eps*m_batches))/l + p )
  a_val.append(int( a) )

print("exp(-lambda(a-p))<= 1/(2*m*eps) -> a_val = ",a_val)

lambda_ = []
a_val =[]

for i in [0,1]:
  l=A_estimatedM[i] * np.minimum(2.0, c_estimatedM[i]) / (2*c_estimatedM[i])
  lambda_.append(l)

  a= np.ceil(- np.log(1/(m_batches))/l + p )
  a_val.append(int( a) )

print("exp(-lambda(a-p))<= 1/m -> a_val = ",a_val)

################################################

print("\nOLR")
print("Info: instead of N_train in a_search, I'm using N-(m_test+1)*a")

A_estimated = 0.6975916
c_estimated =  4.851
lambda_=A_estimated * np.minimum(2.0, c_estimated) / (2*c_estimated) 
N = 3546

m_test = 18 # change that if you like
 
a_search=lambda a : lambda_*(a-p) + np.log(0.025*a/(2*eps*(N-(m_test+1)*a))) 

#a_val computed

a_val= int(np.ceil(newton(a_search,1))) # 1=a_0, starting point

print("exp(-lambda(a-p))<= delta/(2*m*eps) -> a_val = ",a_val)

###

a_search=lambda a : lambda_*(a-p) + np.log(a/(2*eps*(N-(m_test+1)*a))) 

#a_val computed

a_val= int(np.ceil(newton(a_search,1))) # 1=a_0, starting point

print("exp(-lambda(a-p))<= 1/(2*m*eps) -> a_val = ",a_val)

###

a_search=lambda a : lambda_*(a-p) + np.log(a/((N-(m_test+1)*a))) 

#a_val computed

a_val= int(np.ceil(newton(a_search,1))) # 1=a_0, starting point

print("exp(-lambda(a-p))<= 1/m -> a_val = ",a_val)

###############################################################################

print("\n\nlambda = A")
print("GAU & NIG")

A_estimatedM=[3.840956,3.868912]#,]#,[3.840956]#
c_estimatedM=[1,1]
m_batches=1000
eps=3.
p=1

###

lambda_ = []
a_val =[]

for i in [0,1]:
  l=A_estimatedM[i]
  lambda_.append(l)

  a= np.ceil(- np.log(0.025/(2*eps*m_batches))/l + p )
  a_val.append(int( a) )

print("exp(-lambda(a-p))<= delta/(2*m*eps) -> a_val = ",a_val)

###

lambda_ = []
a_val =[]

for i in [0,1]:
  l=A_estimatedM[i]
  lambda_.append(l)

  a= np.ceil(- np.log(1/(2*eps*m_batches))/l + p )
  a_val.append(int( a) )

print("exp(-lambda(a-p))<= 1/(2*m*eps) -> a_val = ",a_val)

lambda_ = []
a_val =[]

for i in [0,1]:
  l=A_estimatedM[i]
  lambda_.append(l)

  a= np.ceil(- np.log(1/(m_batches))/l + p )
  a_val.append(int( a) )

print("exp(-lambda(a-p))<= 1/m -> a_val = ",a_val)

################################################

print("\nOLR")
print("Info: instead of N_train in a_search, I'm using N-(m_test+1)*a")

A_estimated = 0.6975916
c_estimated =  4.851
lambda_=A_estimated
N = 3546

m_test = 18 # change that if you like
 
a_search=lambda a : lambda_*(a-p) + np.log(0.025*a/(2*eps*(N-(m_test+1)*a))) 

#a_val computed

a_val= int(np.ceil(newton(a_search,1))) # 1=a_0, starting point

print("exp(-lambda(a-p))<= delta/(2*m*eps) -> a_val = ",a_val)

###

a_search=lambda a : lambda_*(a-p) + np.log(a/(2*eps*(N-(m_test+1)*a))) 

#a_val computed

a_val= int(np.ceil(newton(a_search,1))) # 1=a_0, starting point

print("exp(-lambda(a-p))<= 1/(2*m*eps) -> a_val = ",a_val)

###

a_search=lambda a : lambda_*(a-p) + np.log(a/((N-(m_test+1)*a))) 

#a_val computed

a_val= int(np.ceil(newton(a_search,1))) # 1=a_0, starting point

print("exp(-lambda(a-p))<= 1/m -> a_val = ",a_val)