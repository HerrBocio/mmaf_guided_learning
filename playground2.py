import numpy as np


#delta = 0.025
#delta=1/.025
#print(1/delta)
m=1000
#print(np.log(1/delta))
#print(1/np.sqrt(m))
#print(np.log(1/delta)/np.sqrt(m))


def func(x):
    return [x,5+x]
t = 0
s=0
N=3

for x in range(3):
    p,q = func(x)
    t +=p
    s += q

print(t/N)
print(s/N)