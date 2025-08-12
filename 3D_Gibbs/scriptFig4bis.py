# from scipy.io import loadmat
# from scipy.spatial.distance import cdist
# from scipy.linalg import cholesky

import matplotlib.pyplot as plt
from numpy.random import default_rng


# import pandas as pd
# import csv
from datetime import datetime

import numpy as np
import numpy.matlib

import tensorflow as tf

# from seaborn import heatmap,color_palette as col

from STOUleo import STOU, variogram, estimation, modello_teoricoSTOU, modello_teoricoMSTOU
# from range3d_k import predict_k_slice
from range3d_k import predict_slice

# from samplerShift import get_gaussian_sampler, Sampler
# from drawRMSE import drawRMSE
# from averRMSE import averRMSE

import time
import os
import h5py
import netCDF4 as nc
from datetime import datetime, timezone


# ------------------------------------------------------------------------------------------------------------
seed = 3  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rng = default_rng(seed)

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
data = tf.cast(data[:, 53:73, 53:73], tf.float64) #per OLR
#data = tf.cast(data[:, :21, :21], tf.float64)

# estimated lambda = popt.x[0]
# STOU
decay = (popt.x[0]*min(1, 1./popt.x[1]) *
         min(1, popt.x[1]/np.sqrt(2)))/(np.sqrt(6*(1+popt.x[1] ** 2)))
#decay = (popt.x[0]-3)/2

# ------------------------------------------------------------------------------------------------------------

# Inizio del codice per creare previsione 1 time ahead
data_path = "datasets/"

eps = [2.99]

ids = [0]


date = datetime.now()
path = 'results/'
name = "Prova_Pratica"

colors = ['orange', 'crimson']

p = [1]  # ,2,3]#,2,3]
k = 1

if not os.path.exists(path):
    os.makedirs(path)

hf = h5py.File(path + name + str(date) + '.h5', 'w')

for el in p:
    for current_id in ids:  # range(datanum):
        max_val = int(data.shape[0]-k)  #
        data_val = data[max_val:, :, :]
        data = data[:max_val, :, :]  # -2000

        # x=hf.create_group(str(datasetsON[current_id])+str(el))

    ground_truth = data_val[k-1, :, :]

    N, x_size, y_size = data.shape
    # c = 1
    # initialize model to estimate parameters, which are independent of the position and
    # use set A, c, lambda to previously estimated values

    Z = STOU(tf.constant([[3, 3]]),
             data, lambda_=decay, c_=popt.x[1], h_t=1)

    Z.N = N

    time_start = time.perf_counter()

    outmin = []
    outmax = []
    out_025 = []
    out_05 = []
    out_25 = []
    out_75 = []
    out_95 = []
    out_975 = []

    data_current = data[:, -el:]  # form pth last column to the end

    # print("p: ", el)
    # print("p: ",el,file=outfile)
    cont = 0
    Z.N = N
    Z.p = el
    # flag=0 per STOU e flag=1 per MSTOU
    Z.estimate_a_k(p=Z.p, eps=eps, flag=0)
    #Z.a=4
    print(Z.a)
    # this is not the k of the k-slices ahead prediction
    Z.k = 1
    # Z.set_model_for_benchmark(el)
    print('computing benchmark')
    # benchmarkA=bench(Z,data,x_size,el,c)

    [output, omin, omax, draws] = predict_slice(
        Z, data, x_size, y_size, 'absB', el, c=popt.x[1], Ndraws=50, eps=eps, mean=0, var=1)
    # print('\n', datasetsON[current_id], 'eps', eps[j], 'flag: ',flag, "a: ", Z.a, "k: ", Z.k, "m: ", Z.m, 'l', Z.lambda_)

    time_elapsed = (time.perf_counter() - time_start)

    print("time: ", time_elapsed)

    outmean = np.mean(output, axis=2)

    out_25.append(np.quantile(output, 0.25, axis=2))
    out_75.append(np.quantile(output, 0.75, axis=2))
    out_05.append(np.quantile(output, 0.05, axis=2))
    out_95.append(np.quantile(output, 0.95, axis=2))
    out_025.append(np.quantile(output, 0.025, axis=2))
    out_975.append(np.quantile(output, 0.975, axis=2))

    # a IS STORED WRT. FLAGS
    group = hf.create_group("LeoLori")
    group.create_dataset('a', data=Z.a)
    group.create_dataset('k', data=k)
    group.create_dataset('p', data=el)
    group.create_dataset('c', data=popt.x[1])
    group.create_dataset('decay', data=decay)
    group.create_dataset('draws', data=output)
    group.create_dataset('min', data=outmin)
    group.create_dataset('max', data=outmax)
    group.create_dataset('q25', data=out_25)
    group.create_dataset('q75', data=out_75)
    group.create_dataset('q025', data=out_025)
    group.create_dataset('q975', data=out_975)
    group.create_dataset('test', data=ground_truth)

    hf.close()

