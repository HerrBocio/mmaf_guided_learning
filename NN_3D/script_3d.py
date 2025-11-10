import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'#,2,3'#,2,3'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
from numpy.random import default_rng
import netCDF4 as nc
from scipy.optimize import newton

from STOU import *
from variogram import *
from SGE3d_EC import Optimization
import numpy as np
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
import jax.numpy as jnp

data_path='/afs/tu-chemnitz.de/project/calibration/'

path='/LOCAL/prol/3d/results/'



normalize_data=False
use_different_eps=False

#datasets=["Gaudiamonddata1A4","NIGdiamonddata1A4L","Gaudiamonddata10","NIGdiamonddata10","NIGdiamonddataBis1","NIGdiamonddata1Long"]#]
#datasetsM=['NIGdiamonddata1A4mln','Gaudiamonddata1A4mln']
#datasetsM=['Trappo']



lr=0.01
h_t=0.05
eps=5
#a=[47,45,47,47,134,207]#47#3
center_pixel=100
p=1

seed = 3  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rng = default_rng(seed)

datasets=['OLR']
file_path = data_path+"OLR_full3d.nc"
olr = nc.Dataset(file_path, mode="r").variables
olr = olr['olra'][:, :, : ]
std_fill=-9.96921e+36

olr =olr.filled(fill_value=min(-100,np.amin(olr)))

print('shape',olr.shape,np.amax(olr),np.amin(olr))

"""
with h5py.File('datasetGauSep1.01.02.02.08.h5', 'r') as f:
    dataset = f['data']
    data = dataset[:]
"""

data = np.array(olr)
data = data[:,:72,:72] #per OLR
print(data.shape)

mean=np.mean(data)

#data=data-mean

print('shape',data.shape,np.amax(data),np.amin(data))

M_variogram, h_estimation = variogram(data)

model = "STOU"
LSEtype = "OLS"

moments=1

# Initial guess for parameters
#initial_guess = np.array([1.0, 1.0, 1.0])
#bounds_STOU = [(0, None)] * len(initial_guess)  # Tutti i parametri > 0
initial_guess = np.array([4.0, 2.0, 2.0, 2.0, 2.0])
bounds_MSTOU = [(3+6*(1+1./moments)**2, None), (0.1, None), (0.1, None), (0.1, None),
                 (0.1, None)]  # theta[0] > 3

popt = estimation(model, LSEtype, M_variogram,
                  h_estimation, initial_guess, bounds_MSTOU)

print('estimated parameters:', popt.x)

#data = tf.cast(data[:, :21, :21], tf.float64)

# estimated lambda = popt.x[0]
# STOU
#decay =(popt.x[0]*min(1, 1./popt.x[1]) *  min(1, popt.x[1]/np.sqrt(2)))/(np.sqrt(6*(1+popt.x[1] ** 2)))

#MSTOU

decay = (popt.x[0]-3)/2


lambda_estimated=[decay]
c_estimated=[int(popt.x[1])]
#lambda_ estimation
print(c_estimated)




#data = data[:, 30:40 , 68:78] #per OLR




def get_simulated_data(filename):
  data=loadmat(filename+'.mat')
  data=data["data"]
  return data

def rescalingU(d,eps=0):
  m=np.nanmin(d)
  M=np.nanmax(d)
  print(M,m)
  p=(1-2*eps)/(M-m)
  q=(M*eps-(1-eps)*m)/(M-m)
  return d*p+q,p,q

def rescalingInv(d,slope,q,eps=0):
  
  #print(d,slope,q)
  #m=np.amin(d)
  #M=np.amax(d)
  #p=(1-2*eps)/(M-m)
  #q=(M*eps-(1-eps)*m)/(M-m)
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


filename=''

shard_size=[1] # it currently must be a proper divisor of x_size-2*p !!!



delta=2/.05
rhoScaling=1./2			

inp=9
#c=1
##m=range(mincones,maxcones+stepcones,stepcones)
#a= range(mina,maxa,stepa)
output=[]
bound=[]
archs=[[300,300,1]]#[30,30,1],[100,100,1],

#m_batches=
#m_test= 
a_val=[5,5] #tbr
Ndraws=30

piScalingLabel=list(range(10,230,20))

piRescaling=[1./10,1./30,1./50,1./70,1./90,1./110,1./130,1./150,1./170,1./190,1./210]#,1./40,1./50]
piRescaling= np.log(piRescaling)
print(piRescaling)

pretraining=[False]

Epoch=100

rescaling=True
for l_,boolean in enumerate(pretraining): 

            
      if boolean:
          Epochs=[150]#,50]
          pretraining_labels=['_preT']
      else:
          Epochs=[60]#,50]
          pretraining_labels=['']
            
      for i,arch in enumerate(archs): # reversed
        print(arch,dim([inp,*arch]))
        for current_id in range(len(datasets)):
          print(datasets[current_id]) 
          #data=get_simulated_data(data_path+datasets[current_id] ) #might be converted into JNP
          #data=data[:Ncoords,:]
          #data,slope,q=rescalingU(data,eps=0.000001)
          #print(slope,q)
          #print(data.shape)
          Ncoords = 8 + 2*int(np.floor(c_estimated[current_id])*p) 
          #test=data[-1,:,:]    
          N=data.shape[0]
          x1_size=data.shape[1]
          x2_size=data.shape[2]
          print(x1_size,x2_size,Ncoords)
          N_train=int(N/3*2)
          a_search=lambda x : lambda_estimated[current_id]*h_t*(x-p) + np.log(0.025*x/(2*eps*N_train)) 
        
          a_val= int(np.ceil(newton(a_search,1)))

          m=int(N_train//a_val)
          m_test=int((N-N_train)//a_val)
          print('newton',a_val)
          inp=int(np.sum([(np.floor(2*((c_estimated[current_id]*el)/np.sqrt(2)+1))+1)**2 for el in range(1,p+1)]))
          print('inp',inp)
          #print(data[10,:,:])
          for k,pir in enumerate(piRescaling):
            print('prior=',pir)
            pathPrior=path+'prior'+str(piScalingLabel[k])+'var/'
            if not os.path.exists(pathPrior):
                os.makedirs(pathPrior)
                print("folder created")
        
            file_ = h5py.File(pathPrior+'11_04_3d_relu'+pretraining_labels[l_] +str(dim([inp,*arch])) +'_' +str(datasets[current_id]) +'_a' +str(a_val) +'_pir' +str(piScalingLabel[k]) +'_m' +str(m) +'_Epoch_'+str(Epochs[l_])+'.h5','w') #+'maxit_'+str(maxit)
            file_m=file_.create_group('m'+str(m))
            print('m=',m)
            #file_data=file_net.create_group(str(datasets[current_id]))
            #print(m)
            #for pir in piRescaling:
            #print(1./pir)
            #file_pir=file_data.create_group('pir'+str(pir))
            #print(file_pir.keys())
            Z = STOU(0,data,lambda_estimated[current_id],c_estimated[current_id],arch,N,m_test-1,m,a_val,p,h_t=0.05)
            Optimization(file_m,Z,boolean,rescaling,eps,delta,data,inp,p,c_estimated[current_id],arch,dim([inp,*arch]),Ndraws,m,m_test,Ncoords,shard_size[i], lr, rhoScaling, epochs=Epoch, piScaling=pir)
                
            file_.close()

