import os
import torch.nn as nn
import torch
import netCDF4 as nc

import torch.optim as optim
from torch import vmap


import jax

import jax.numpy as jnp


import numpy as np
#import matplotlib.pyplot as plt



data_path="../datasets/" #/LOCAL/prol/
path='/LOCAL/prol/LSTM/'

#path='/afs/tu-chemnitz.de/project/calibration/debug/'
def get_simulated_data(filename):
    data=loadmat(filename+'.mat')
    data=data["data"]
    return data

def makeh5(net,hdata,names):

  '''
  Takes as input the h5 file pointer, data to be stored and the respective names
  Creates and stores data structures in an (already defined!) h5 group
  '''
  for j in range(len(names)):
    net.create_dataset(names[j],data=hdata[j])

def crps_univ_rank_mapped_torch(y, x):
    """
    y: scalar or tensor of shape [1] or [M]
    x: tensor of shape [M]
    """

    M = x.shape[0]

    # First term: mean absolute error between x and y
    term1 = torch.sum(torch.abs(x - y)) / M

    # Second term: double sum |x_i - x_j|
    # Equivalent to broadcasting in PyTorch
    diff_matrix = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    double_sum = diff_matrix.sum()

    crps = term1 - double_sum / (2 * M**2)
    return crps

def rmse_univ(y,x):
    return torch.mean((x-y)**2)




####################################################################################
# This script launches the optimization routine for synthetic data GauA4 and NIGA4
# The hyperparameters A,c,lambda_ are already estimated

gpu=0

torch.device('cuda:'+str(gpu))
cuda='cuda:'+str(gpu)
seeds=[5,7,11,13,15,17,19]







'''

path='/LOCAL/prol/OLR_full/natale/' #/afs/tu-chemnitz.de/project/calibration/OLR/results/'
if not os.path.exists(path):
    os.makedirs(path)
    print("folder created")
'''

file_path = '/afs/tu-chemnitz.de/project/calibration/OLR_full.nc'#Almut_plusFuture.nc'

olr = nc.Dataset(file_path, mode="r").variables
#print(olr)
olr=olr['olra'][26:,:]#,:]
print(olr.shape)
std_fill=-9.96921e+36

olr = np.transpose(np.mean(olr.filled(fill_value=min(-100,np.amin(olr))),axis=1))#np.nanmean(olr)
print('f')
print('shape',olr.shape,np.amax(olr),np.amin(olr))


olr_detrended= olr #-trend


print(olr_detrended.shape)
  

#olr_detrended=np.array(list(olr_detrended))

print(olr_detrended[:,0].max(),olr_detrended[:,0].min())

#olr_detrended=np.array( [olr[el,:]-mean[el] for el in range(len(mean))])




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


#initialization
olr_detrended=olr_detrended[:,:]


datasetsM= ['OLR_full']#'Gaudiamonddata1A4mln'

#np.save('~/Desktop/ffnn/datasets/'+datasetsM[0]+'/'+'flow.npy',olr_detrended)

A_estimated=hatA#,[3.840956]#
c_estimated=hatc

p=1
c=int(np.floor(hatc))

#size = c*p+1

a_val=64

Ndraws=30
#from tqdm import trange
def get_simulated_data(filename):
    data=loadmat(filename+'.mat')
    data=data["data"]
    return data

def create_folder(new_path):
  '''
  Creates directory 'new_path'
  '''
  if not os.path.exists(new_path):
    os.makedirs(new_path)
    print("folder created")


data_path='../Desktop/ffnn/datasets/'

path='dataset/'


for current_id in range(len(datasetsM)):
	
	create_folder(path+datasetsM[current_id]+'')
	
	data= olr_detrended[:,:] 
	
	print('dataschape del dato creato',data.shape)
	create_folder(path+datasetsM[current_id])
	np.save(path+datasetsM[current_id]+''+'/'+'data.npy',data)
	