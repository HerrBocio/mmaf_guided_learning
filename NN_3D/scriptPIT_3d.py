import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2'#,2,3'#,2,3'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
from numpy.random import default_rng
import netCDF4 as nc

from STOUpozo import STOU
from SGE3dPIT import OptSGD
import numpy as np
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
import jax.numpy as jnp
from STOUleo import variogram,estimation, modello_teoricoSTOU, modello_teoricoMSTOU

data_path="datasets/"
path='Results/'
normalize_data=False
use_different_eps=False

#datasets=["Gaudiamonddata1A4","NIGdiamonddata1A4L","Gaudiamonddata10","NIGdiamonddata10","NIGdiamonddataBis1","NIGdiamonddata1Long"]#]
#datasetsM=['NIGdiamonddata1A4mln','Gaudiamonddata1A4mln']
#datasetsM=['Trappo']


redux=200000
Ndraws=50
Ncones= 500 #how many cones embedded for each batch
mincones=50
maxcones=500
stepcones=50

mina=5
maxa=105
stepa=10

Ncoords=201
lr=0.01
h_t=0.05
eps=5
#a=[47,45,47,47,134,207]#47#3
a=[3,65]
center_pixel=100
p=[1]

seed = 3  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rng = default_rng(seed)

datasets=['OLR']
file_path = r"OLR.nc"
olr = nc.Dataset(file_path, mode="r").variables
data = olr['olra'][:, :, 36:109]
"""
with h5py.File('datasetGauSep1.01.02.02.08.h5', 'r') as f:
    dataset = f['data']
    data = dataset[:]
"""

data = np.array(data)

M_variogram, h_estimation = variogram(data)

model = "STOU"
LSEtype = "OLS"
# Initial guess for parameters
initial_guess = np.array([1.0, 1.0, 1.0])
bounds_STOU = [(0, None)] * len(initial_guess)  # Tutti i parametri > 0
#initial_guess = np.array([4.0, 2.0, 2.0, 2.0, 2.0])
bounds_MSTOU = [(3, None), (0.1, None), (0.1, None), (0.1, None),
                 (0.1, None)]  # theta[0] > 3

popt = estimation(model, LSEtype, M_variogram,
                  h_estimation, initial_guess, bounds_STOU)

print('estimated parameters:', popt.x)
data = data[:, 43:73, 43:73] #per OLR
#data = tf.cast(data[:, :21, :21], tf.float64)

# estimated lambda = popt.x[0]
# STOU
decay = (popt.x[0]*min(1, 1./popt.x[1]) *
         min(1, popt.x[1]/np.sqrt(2)))/(np.sqrt(6*(1+popt.x[1] ** 2)))
#decay = (popt.x[0]-3)/2

lambda_estimated=[decay]
c_estimated=[1]#popt.x[1]]
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

shard_size=3 # it currently must be a proper divisor of x_size-2*p !!!

delta=2/.05
rhoScaling=1./2			

inp=9
##m=range(mincones,maxcones+stepcones,stepcones)
#a= range(mina,maxa,stepa)
output=[]
bound=[]
archs=[[10,10,1]]#,[10,1]]#,[10,10,1],[11,10,10,10,10,10,1],[60,50,30,20,10,1]] #[10,10,1],

m=15#[5,10,25,50]#[50,70,85,100]#
piRescaling=range(10,2*m,10)

#file_=h5py.File(path+filename+'elpitPi50k.h5','w')#+str(datetime.now().strftime('%m_%d_%H_%M'))   

for i,arch in enumerate(reversed(archs)):#reversed
  print(arch)
  #file_net=file_.create_group(str(dim([inp,*arch])))
  for current_id in range(len(datasets)):
    print(datasets[current_id])
    #print(data[10,:,:])
    #data=  jnp.reshape(jnp.arange(12100),(100,11,11))             #get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
    #data=data[:,-redux:]
    data,slope,q=rescalingU(data,eps=0.000001)
    #print(slope,q)
    print(data.shape)
    test=data[-1,:,:]    
    N=data.shape[0]
    x1_size=data.shape[1]
    x2_size=data.shape[2]
    #print(data[10,:,:])
    
    #file_data=file_net.create_group(str(datasetsM[current_id]))
    #print(m)
    #for pir in piRescaling:
    #print(1./pir)
    #file_pir=file_data.create_group('pir'+str(pir))
    #print(file_pir.keys())
    Z = STOU(0,data,lambda_estimated[current_id],c_estimated[current_id],arch,N,m,a[0],p[0],h_t=0.05)
    out,pars,b,pit,batch,ks= OptSGD(Z,x1_size,x2_size,'_',eps,delta,data,inp,p[0],c_estimated[current_id],arch,dim([inp,*arch]),Ndraws,m,Ncoords,shard_size,lr,rhoScaling,slope,q,1/15)
    outIR=rescalingInv(out,slope,q,eps=0.000001) 
    #print(outIR.shape,pars,b)   
    hdata=[out,outIR,b,pars,pit,test,np.array(batch),ks.pvalue]
    names=['output','outIR','bound','pars','pit','test','batch']
    #makeh5(file_pir,hdata,names,path) # SET BACK!!!

#file_.close()

