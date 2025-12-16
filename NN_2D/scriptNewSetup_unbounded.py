import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'#,2,3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

from STOU import STOU
from params import *
from SGE_unbounded import Optimization
from utils import *
import numpy as np
import h5py
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
data_path="../datasets_2D/" #/LOCAL/prol/
#data_path="datasets_2D/" #/LOCAL/prol/

os.makedirs('/LOCAL/jasst/results/nopreT', exist_ok=True)
path='/LOCAL/jasst/results/nopreT/'
#path='/afs/tu-chemnitz.de/project/calibration/debug/'


####################################################################################
# This script launches the optimization routine for synthetic data GauA4 and NIGA4
# The hyperparameters A,c,lambda_ are already estimated


ids=[1]
A_estimatedM=[3.840956,3.868912]#,]#,[3.840956]#
c_estimatedM=[1,1]

m_test= 101

Ndraws=30

Ncoords=10 #max201
lr=0.005

eps=3.
p=1

lambda_=[]
a_val=[]

#lambda_ estimation
for i in range(len(datasetsM)):
  l=A_estimatedM[i] * np.minimum(2.0, c_estimatedM[i]) / (2*c_estimatedM[i])
  print(l)
  lambda_.append(l)

  a= np.ceil(- np.log(0.025/(2*eps*m_batches))/l + p )
  a_val.append(int( a) )

print('a val',a_val,lambda_)

center_pixel=5


shard_size= [8] #[8,2,1]#,2,8]#,1]#,33]#99,?

#sets the variance of the reference distribution 
piScalingLabel=list(range(10,230,20))
piRescaling=[1./10, 1./30, 1./50, 1./70, 1./90,1./110, 1./130, 1./150, 1./170, 1./190, 1./210] #   
piRescaling= np.log(piRescaling)
print(piRescaling)

pretraining=[False]

rescaling=False

day='151225'
filename=day+'_full_relu_std'

for l_,boolean in enumerate(pretraining): 
  #loops over pretraining choice
  if boolean:
      Epochs=150
      pretraining_labels='_preT'
  else:
      Epochs=epochs_nopreT #60
      pretraining_labels=''
  for i,arch in enumerate(archs): # reversed
    #loops over architectures
    print(arch,dimComp([inp,*arch]))
    for current_id in range(len(datasetsM)):
      #loops over datasets
      print(datasetsM[current_id]) 
      data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
      data=data[:Ncoords,:]
      N=data.shape[1]
      x_size=data.shape[0]
      c=1
      for k,pir in enumerate(piRescaling):
        #loops over reference distributions
        print('prior=',pir)
        pathPrior=path+'prior'+str(piScalingLabel[k])+'var/'
        create_folder(pathPrior)
        file_ = h5py.File(pathPrior+day+filename+pretraining_labels +str(dimComp([inp,*arch]))+'_' +str(datasetsM[current_id]) +'_a' +str(a_val) +'_pir' +str(piScalingLabel[k]) +'_m' +str(m_batches) +'_Epoch_'+str(Epochs)+'.h5','w')
        file_m=file_.create_group('m'+str(m_batches))
        print('m=',m_batches)
        Z = STOU(A_estimatedM[current_id],c_estimatedM[current_id],arch,N-m_test*a_val[current_id],m_test-1,m_batches,a_val[current_id],p,h_t[0])
        Optimization(file_m,Z,x_size,boolean,rescaling,eps,delta,data,inp,p,c,arch,dimComp([inp,*arch]),Ndraws,m_batches,m_test,Ncoords,shard_size[0], lr, epochs=Epochs, piScaling=pir)
        file_.close()