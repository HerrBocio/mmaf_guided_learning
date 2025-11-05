import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'#,2,3'#,2,3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

from STOUNewSetup import STOU
from SGEmultiEC import Optimization
import numpy as np
import h5py
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import chisquare#kstest as kstest
from scipy.stats import randint
from scipy.stats import kstest
import os
print(os.getcwd())


data_path="datasets_2D/" #/LOCAL/prol/
#data_path="../datasets_2D/" #/LOCAL/prol/
path='/LOCAL/prol/s_data/results/preT'

#path='/afs/tu-chemnitz.de/project/calibration/s_data/results/'


normalize_data=False
use_different_eps=False

if not os.path.exists(path):
    os.makedirs(path)
    print("folder created")

def get_simulated_data(filename):
  data=loadmat(filename+'.mat')
  data=data["data"]
  return data

def rescalingU(d,eps=0):
  m=np.amin(d)
  M=np.amax(d)
  p=(1-2*eps)/(M-m)
  q=(M*eps-(1-eps)*m)/(M-m)
  return d*p+q,p,q

def rescalingInv(d,slope,q,eps=0):

  return (d - q)/slope

def dim(net):
  d=0
  for i in range(len(net)-1):
    d+=(net[i]+1)*net[i+1]
  return d

def makeh5(net,hdata,names,path):
	
  #for i in range(len(datasets)):
  #x=net.create_group(str(dataset))
  for j in range(len(names)):
    net.create_dataset(names[j],data=hdata[j])
  #file_.close()


datasetsM= ['Gaudiamonddata1A4','NIGdiamonddata1A4']#
ids=[0]
A_estimatedM=[3.840956,3.868912]#,]#,[3.840956]#
c_estimatedM=[1,1]
m_batches=[1000] #range(1000,11000,1000)#[3000,4000,5000,6000,7000,8000,9000,10000]##[

Ndraws=30

Ncoords=10 #max201
lr=0.005
h_t=[1]
eps=3.
p=1

lambda_=[]
a_val=[]

for i in range(len(datasetsM)):
  l=A_estimatedM[i] * np.minimum(2.0, c_estimatedM[i]) / (2*c_estimatedM[i])
  print(l)
  lambda_.append(l)

  a= np.ceil(- np.log(0.025/(2*eps*m_batches[0]))/l + p )
  a_val.append(int( a) )

print('a val',a_val,lambda_)




center_pixel=5



filename=''

delta=1/.025
rhoScaling=1./2			
acrit=.75
inp=3

output=[]
bound=[]
archs=[[800,800,800,1]]#[[30,30,1],[100,100,1],[300,300,1]]  ##,,[300,300,1],[100,100,1],
shard_size= [1] #[8,2,1]#,2,8]#,1]#,33]#99,?



#piRescaling=range(10,60,10)
#piRescaling=[1,*piRescaling]
#piRescaling=[-1,-2,-4,-6,-8]#,10,20]
piScalingLabel=list(range(110,230,20))

piRescaling=[1./110, 1./130, 1./150, 1./170, 1./190, 1./210] #  1./10, 1./30, 1./50, 1./70, 1./90, 
piRescaling= np.log(piRescaling)
print(piRescaling)

pretraining=[True]

if pretraining[0]:
    Epochs=[150]#,50]
    pretraining_labels=['_preT']
else:
    Epochs=[60]#,50]
    pretraining_labels=['']

rescaling=False
#m_batches=[500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
m_test= 101




for l_,boolean in enumerate(pretraining): 

      for i,arch in enumerate(archs): # reversed
        print(arch,dim([inp,*arch]))
        for current_id in ids: #range(len(datasetsM)):
              print(datasetsM[current_id]) 
              data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
              data=data[:Ncoords,:]
              #gtruthT=data[:,-1]
              #gtruthV=data[:,-Ndraws*a-1]
              
              #data,slope,q=rescalingU(data,eps=0.000001)
              
      
              
              N=data.shape[1]
              x_size=data.shape[0]
              c=1
            
              for k,pir in enumerate(piRescaling):
                print('prior=',pir)
                pathPrior=path+'prior'+str(piScalingLabel[k])+'var/'
                if not os.path.exists(pathPrior):
                    os.makedirs(pathPrior)
                    print("folder created")
                for m in reversed(m_batches): #reversed
                    file_ = h5py.File(pathPrior+'10_12_full_relu_std'+pretraining_labels[l_] +str(dim([inp,*arch])) +'_' +str(datasetsM[current_id]) +'_a' +str(a_val[current_id]) +'_pir' +str(piScalingLabel[k]) +'_m' +str(m) +'_Epoch_'+str(Epochs[l_])+'.h5','w') #+'maxit_'+str(maxit)
                    file_m=file_.create_group('m'+str(m))
                    print('m=',m)
          
                    Z = STOU(0,data,A_estimatedM[current_id],c_estimatedM[current_id],arch,N-m_test*a_val[current_id],m_test-1,m,a_val[current_id],p,1,h_t[0])
                      
                    Optimization(file_m,Z,x_size,boolean,rescaling,eps,delta,data,inp,p,c,arch,dim([inp,*arch]),Ndraws,m,m_test,Ncoords,shard_size[i], lr, rhoScaling, epochs=Epochs[l_], piScaling=pir, acrit=acrit)
                    
                    file_.close()
  
  
  
