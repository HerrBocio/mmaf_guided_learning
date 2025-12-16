import numpy as np


#delta = 0.025
#delta=1/.025
#print(1/delta)
m=1000
#print(np.log(1/delta))
#print(1/np.sqrt(m))
#print(np.log(1/delta)/np.sqrt(m))


el = np.array([[1,2,7],[3,4,8],[5,6,9]])
x= np.array([1,5])
y=np.matmul(x,el[:-1,:]) +el[-1,:]


#print(el[:-1,:])
#print(el[-1,:])
#print(y)


el = np.array([[1,3,5],[2,4,6],[7,8,9]])
x= np.array([1,5])
y=np.matmul(x,el[:-1,:]) +el[-1,:]

print(el[:-1,:])
print(el[-1,:])
print(y)