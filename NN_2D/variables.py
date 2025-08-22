import numpy as np

datasetsM=['NIGdiamonddata1A4','Gaudiamonddata1A4'] #j the datasets we both predict on
A_estimatedM=[3.840956,3.868912]	
c_estimatedM=[1,1] #j actually, the estimated c's would be <1, but then the cones would be empty, so we work with c=1, which are the real values of c (with which the datasets had been created)
redux=100000 #j we will cut out all the data from before that timestamp
h_t=0.05
eps=5
a=[3,3] #j here, a is chosen equal to 3 for both datasets. normally, one would choose a like in Remark 3.9
center_pixel=100 #j not relevant (?)
p=[1] #j one could try different values for p. if one does, inp has to be changed!!
inp=3 #j equals a(p,c) from the paper. has to be changed if p or c change #i write a function that computes inp
m=30#[5,10,25,50]#[50,70,85,100]#
piRescaling=range(10,int(np.floor(3.5*m)),10) #j gridsearch for prior. the mean of the prior is always zero, these are different values for the variance
archs= [[1]] #j only working with linear predictors


def rescalingU(d,eps=0):
  """#j
  the procedure works best if the values of the dataset are rescaled in such a way that they all lie between 0 and 1 (??)

  d: data
  eps: ???
  """
  m=np.amin(d)
  M=np.amax(d)
  p=(1-2*eps)/(M-m)
  q=(M*eps-(1-eps)*m)/(M-m)
  return d*p+q,p,q

def rescalingInv(d,slope,q,eps=0):
  """"#j
  reverses rescalingU

  d: data
  slope: ???
  q: ???
  eps: ???
  """
  
  #print(d,slope,q)
  #m=np.amin(d)
  #M=np.amax(d)
  #p=(1-2*eps)/(M-m)
  #q=(M*eps-(1-eps)*m)/(M-m)
  return (d - q)/slope

def dim(net):
  '''
  returns number of parameters in the network
  '''
  d=0
  for i in range(len(net)-1):
    d+=(net[i]+1)*net[i+1]
  return d