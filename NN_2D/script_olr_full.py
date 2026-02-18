import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'#,2,3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

from STOU import STOU
from SGE import Optimization
from utils import *
import numpy as np
import h5py
import netCDF4 as nc
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import newton
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize

def makeh5(net,hdata,names,path):
	
  #for i in range(len(datasets)):
  #x=net.create_group(str(dataset))
  for j in range(len(names)):
    net.create_dataset(names[j],data=hdata[j])
  #file_.close()

def dim(net):
  d=0
  for i in range(len(net)-1):
    d+=(net[i]+1)*net[i+1]
  return d



def linear_detrending(data):


	'''
	Spatio-temporal OLS method: it is assumed the spatio-temporal indices are integers spanning from 1 to ns (nt)
	'''

	ns,nt= np.shape(data)
	
	one_t =  lambda row: jnp.array([jnp.sum( row*j )for j in range(nt)])
	
	one_t = jnp.sum(jax.vmap(one_t)(data))

	print('one t', one_t.shape)
  
	tbar= nt*(nt+1)/2

	sbar= ns*(ns+1)/2

	xbar = jnp.mean(data)

	two_t = jnp.sum( jnp.arange(nt)**2)*ns
	
	a_t= - (one_t - ns*nt*xbar*tbar)/(two_t - ns*nt*tbar**2)

	one_s =  lambda row:jnp.array([  jnp.sum(row*i) for i in range(ns)])
	
	one_s = jnp.sum(jax.vmap(one_s)(np.transpose(data)))
	
	two_s = jnp.sum( jnp.arange(ns)**2)*nt

	a_s =  (one_s - ns*nt*xbar*sbar)/(two_s - ns*nt*sbar**2)
	
	b = xbar - a_s * sbar - a_t * tbar

	return a_t,a_s,b



path='/LOCAL/prol/OLR_full/a_4/' #/afs/tu-chemnitz.de/project/calibration/OLR/results/'
if not os.path.exists(path):
    os.makedirs(path)
    print("folder created")

file_path = '/afs/tu-chemnitz.de/project/calibration/OLR_full.nc'#Almut_plusFuture.nc'

olr = nc.Dataset(file_path, mode="r").variables
#print(olr)
olr=olr['olra'][:,:]#,:]
#print(olr.shape)
std_fill=-9.96921e+36

olr = np.transpose(np.mean(olr.filled(fill_value=min(-100,np.amin(olr))),axis=1))#np.nanmean(olr)
print('f')
print('shape',olr.shape,np.amax(olr),np.amin(olr))


#detrending
a_t,a_s,b=linear_detrending(olr[:,:])

print(a_t,a_s,b)

s_trend = lambda s: jnp.ones(olr.shape[-1])*s

s_trend = jax.vmap(s_trend)(jnp.arange(olr.shape[0]))

t_trend = lambda t: jnp.ones(olr.shape[0])*t

t_trend = jnp.transpose(jax.vmap(t_trend)(jnp.arange(olr.shape[-1])))

print('mesh', s_trend.shape,t_trend.shape)
                        
trend = a_t*t_trend + a_s*s_trend + b

print('trend',trend.shape)

mean=np.mean(olr)

olr_detrended= olr #-trend


print(olr_detrended.shape)
  

#olr_detrended=np.array(list(olr_detrended))

print(olr_detrended[:,0].max(),olr_detrended[:,0].min())

#olr_detrended=np.array( [olr[el,:]-mean[el] for el in range(len(mean))])

s=0
bad=[]

'''
for i in range(olr_detrended.shape[1]):
  if (adfuller(olr_detrended[:,i])[1]<5e-2):
    s+=1
  else: bad.append(i) 
print('stationarity',s,bad)

'''



#Y=np.delete(olr_detrended,bad,axis=0)

Y=np.array(list(olr_detrended))
nrY, ncY = Y.shape
print('new dataset',nrY,ncY)


#parameters estimation
c=1
dt=2
dy=10


ndata = nrY * ncY

s1 = np.nansum(Y)
s2 = np.nansum(Y**2)
s3 = np.nansum(Y**3)
s4 = np.nansum(Y**4)

k2 = (1 / (ndata * (ndata - 1))) * (ndata * s2 - s1**2)

d01 = Y.copy()

print(d01.shape)

d01[:, 0:ncY-dt] = d01[:, dt:ncY]
d01[:, ncY-dt:] = np.nan
g01 = np.nanmean((Y - d01)**2) / k2

d10 = Y.copy()
d10[0:nrY-dy, :] = d10[dy:nrY, :]
d10[nrY-dy:, :] = np.nan
g10 = np.nanmean((Y - d10)**2) / k2

print('gg',g01,g10)

hatA = -np.log(1 - g01 / 2) / ( dt)
hatc = -(hatA * dy) / np.log(1 - g10 / 2)

lambda_=hatA * np.minimum(2.0, hatc) / (2*hatc)

print('hat',hatA,hatc,lambda_)


archs= [[30,30,1],[100,100,1]] # [[800,800,800,1]]  #  [300,300,1]]#
#archs= [ [15,15,15,15,15,1],[50,50,50,50,50,1],[150,150,150,150,150,1] ]


shard_size= [8,4] #[1]# 

#initialization
Ncoords=8+2*int(np.floor(hatc))
OLR=olr_detrended[:Ncoords,:]

datasets= ['OLR_full']#'Gaudiamonddata1A4mln'

A_estimated=[hatA]#,[3.840956]#
c_estimated=[hatc]

Ndraws=30

lr=0.005
h_t=1
eps=3.

center_pixel=10
p=1


filename=''

delta=1/.025
rhoScaling=1./2			
acrit=.75
inp=(2*int(np.floor(hatc)))+1

output=[]
bound=[]

lambda_=[]
a_val=[]

#m_batches=[500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
m_batches=[60] #range(1000,11000,1000)#[3000,4000,5000,6000,7000,8000,9000,10000]##[
m_test= 32
Ncones_test=18

#,[30,30,1]]#,[100,100,1],[300,300,1]]#[8,10,8,1]]#,[10,1][8,9,11,10,1],[1],
#,33]#99,?
#lambda_ estimation
'''
for i in range(len(datasets)):
  l=A_estimated[i] * np.minimum(2.0, c_estimated[i]) / (2*c_estimated[i])
  print(l)
  lambda_.append(l)

  a= np.ceil(- np.log(0.025/(2*eps*m_batches[i]))/l + p )
  a_val.append(int( a) )

print('a val',a_val,lambda_)
'''

#piRescaling=range(10,60,10)
#piRescaling=[1,*piRescaling]
#piRescaling=[-1,-2,-4,-6,-8]#,10,20]

piScalingLabel=list(range(10,230,20))

piRescaling=[1./10,1./30,1./50,1./70,1./90,1./110,1./130,1./150,1./170,1./190,1./210]#,1./40,1./50]
piRescaling= np.log(piRescaling)
print(piRescaling)

pretraining=[True]#,True]

rescaling=True



#print('dataset size',OLR.shape)
for l_,boolean in enumerate(pretraining):  
  
  if boolean:
    Epochs=[15000] #range(1,12001,10) #
    Nepochs=15000  #len(Epochs)
    preTlabel='_preT'
  else:
    Epochs=[20000]  #range(1,5001,10) #
    Nepochs=20000   #len(Epochs)
    preTlabel=''
  for Epoch in Epochs:
      for i,arch in enumerate(archs): # reversed
        print(arch)
        for current_id in reversed(range(len(datasets))):
                print(datasets[current_id]) 
                inp=int(np.sum([2*np.floor(c_estimated[current_id]*el)+1 for el in range(1,p+1)]))
                print(datasets[current_id],arch,dim([inp,*arch]),p) 
                Ncoords = 8 + 2*int(np.floor(c_estimated[current_id])*p) 
			
        
                c=int(np.floor(c_estimated[current_id]))
                data=olr_detrended[:Ncoords,:]       

                N=data.shape[1]
                x_size=data.shape[0]
                #a_val=4

                N_train=  N - (Ncones_test+1)*a_val  #3002 # 
			
                lambda_=A_estimated[current_id] * np.minimum(2.0, c_estimated[current_id]) / (2*c_estimated[current_id])    
			
                a_search=lambda x : lambda_*h_t*(x-p) + np.log(0.025*x/(2*eps*N_train)) 
                
                #a_val computed
                
                a_val= int(np.ceil(newton(a_search,1)))

                #print('newton',a_val)

                # a_val fixed
		
                		
		
                print("fixed val for a_t", a_val)

			
                m=N_train//a_val
                m_test= (N - N_train)//a_val  # N_test = 1266         
                print('olr shape',x_size,N,N-m_test*a_val)
                print('a val',a_val,lambda_,m)

                for k,pir in enumerate(piRescaling):
                                print('prior=',piScalingLabel[k])
                                pathPrior=path+'prior'+str(piScalingLabel[k])+'var/'
                                if not os.path.exists(pathPrior):
                                        os.makedirs(pathPrior)
                                        print("folder created")
				#for m in reversed(m_batches): #reversed
                                file_ = h5py.File(pathPrior+'01_18_relu_rescaling'+preTlabel+str(dim([inp,*arch])) +'_' +str(datasets[current_id]) +'_a' +str(a_val) +'_pir' +str(piScalingLabel[k]) +'_m' +str(m) +'_Epoch_'+str(Epochs)+'.h5','w')
                                file_m=file_.create_group('m'+str(m))
                                print('m=',m)		
                                Z = STOU(A_estimated[current_id],c,arch,N-m_test*a_val,m_test-1,m,a_val,p,h_t)
		
                                Optimization(file_m,Z,x_size,boolean,rescaling,eps,delta,OLR,inp,p,c,arch,dimComp([inp,*arch]),Ndraws,m,m_test,Ncoords,shard_size[0], lr,  epochs=Epoch, piScaling=pir)
                                
                                file_.close()

