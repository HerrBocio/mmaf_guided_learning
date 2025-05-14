# import h5py
# from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
from numpy.random import default_rng
from scipy.special import gamma

# import pandas as pd
# import csv
from datetime import datetime

import numpy as np
import numpy.matlib

import tensorflow as tf

# from seaborn import heatmap,color_palette as col

from STOUleo import STOU
from range3d_h import predict_h_slice
from range3d_h import predict_slice

# from samplerShift import get_gaussian_sampler, Sampler
# from drawRMSE import drawRMSE
# from averRMSE import averRMSE

import time
import os

from scipy.optimize import basinhopping  # , minimize
# ------------------------------------------------------------------------------------------------------------
# epsilon = 3 for all these sdimulations
# We can consider a=2 for the best approximation of the Gibbs distribution !!
# Gaussian separable Kernel function
# STOU lambda = 0.48696607
a = 313  # using Remark 3.9
# a = 73  # using Remark 3.15
seed = 18

# STOU lambda = 0.67321802
# a = 253 #using Remark 3.9
# a = 57 #using Remark 3.15
# seed = 264

# STOU lambda = 0.52616688
# a = 297 #using Remark 3.9
# a = 63 #using Remark 3.15
# seed = 6539

# STOU lambda = 0.31874551
# a = 415  # using Remark 3.9
# a = 102 #using Remark 3.15
# seed = 23845


# MSTOU lambda = (5.09891329-3)/2 = 1.04945664
# a = 3318 # using Remark 3.9
# a = 266  # using Remark 3.15
# seed = 79

# MSTOU lambda = (5.69557828-3)/2 = 1.34778914
# a = 2324 #using Remark 3.9
# a = 192 #using Remark 3.15
# seed = 523

# MSTOU lambda = (7.17311579-3)/2 = 2.086557895
# a = 1272 #using Remark 3.9
# a = 112  # using Remark 3.15
# seed = 1727

# MSTOU lambda = (6.23971446-3)/2 = 1.61985723
# a = 1798 #using Remark 3.9
# a = 152 #using Remark 3.15
# seed = 74527

# ==================================

# Matern separable Kernel function
# STOU lambda = 0.18130299
# a = 604  # using Remark 3.9
# a = 157  # using Remark 3.15
# seed = 18

# STOU lambda = 0.46986442
# a = 321 #using Remark 3.9
# a = 75 #using Remark 3.15
# seed = 264

# STOU lambda = 0.43915669
# a = 335 #using Remark 3.9
# a = 79 #using Remark 3.15
# seed = 6539

# STOU lambda = 0.21860162
# a = 533 #using Remark 3.9
# a = 136 #using Remark 3.15
# seed = 23845


# MSTOU lambda = (3.64751328-3)/2 = 0.32375664
# a = 19284 # using Remark 3.9
# a = 1094  # using Remark 3.15
# seed = 79

# MSTOU lambda = (4.93654299-3)/2 = 0.968271495
# a = 3726 #using Remark 3.9
# a = 120 #using Remark 3.15
# seed = 523

# MSTOU lambda = (5.59828711-3)/2 = 1.299143555
# a = 2448 #using Remark 3.9
# a = 201  # using Remark 3.15
# seed = 1727

# MSTOU lambda = (6.42754676-3)/2 = 1.71377338
# a = 1663 #using Remark 3.9
# a = 142 #using Remark 3.15
# seed = 74527


rng = default_rng(seed)

p = [1]  # ,2,3]#,2,3]
h = 3

# Grid dimensions
# ♣ n_t, n_x, n_y = 2001, 31, 31
n_t, n_x, n_y = 2001, 31, 31
'''
# Metodo che fa il dataset tutto in una volta sola
# Spatial covariance parameters
length_scale = 1.0
sigma = 2.0

# Covariance matrix
t_coords = np.arange(n_t)
x_coords = np.arange(n_x)
y_coords = np.arange(n_y)

t, xx, yy = np.meshgrid(t_coords, x_coords, y_coords, indexing='ij')
x = xx[0, :, :]
y = yy[0, :, :]

spatial_coords = np.column_stack([t.ravel(), xx.ravel(), yy.ravel()])
d = cdist(spatial_coords, spatial_coords)
K = sigma**2 * np.exp(-d**2 / (2 * length_scale**2))
#a =q
#K = sigma**2 * np.exp(-ad)*(a**2/3*d**2+a*d+1)
K += 1e-6 * np.eye(K.shape[0])

L = cholesky(K, lower=True)

Z = rng.standard_normal(size=(K.shape[0]))

data = L @ Z
data = data.reshape(n_t, n_x, n_y)
'''
# Metodo che prima fa correlazione temporale e poi spaziale (ugualea quello sopra ma diviso in due)
# Spatial covariance parameters
# inserisco un valore di reshape di scala come nel libro GP for Machine Learning, pag 15
length_scale_space = 4.0
sigma_space = 2.0

# Temporal covariance parameters
# inserisco un valore di reshape di scala come nel libro GP for Machine Learning, pag 15
length_scale_time = 4.0
sigma_time = 2.0


# === Spatial covariance matrix ===
x_coords = np.arange(n_x)
y_coords = np.arange(n_y)
x, y = np.meshgrid(x_coords, y_coords, indexing='ij')
spatial_coords = np.column_stack([x.ravel(), y.ravel()])
d_space = cdist(spatial_coords, spatial_coords)
# shape (n_x*n_y, n_x*n_y)

# Gaussian kernel per componenete spaziale
K_space = sigma_space**2 * np.exp(-d_space**2 / (2 * length_scale_space**2))

# Matern kernel per componenete temporale
# a_space = 1/3.0
# K_space = sigma_space**2 * a_space**3 * np.exp(-a_space*d_space) * \
#    (a_space**2/3*d_space**2+a_space*d_space+1)

K_space += 1e-6 * np.eye(K_space.shape[0])


# === Temporal covariance matrix ===
t_coords = np.arange(n_t).reshape(-1, 1)
d_time = cdist(t_coords, t_coords)
# shape (n_t, n_t)

# Gaussian kernel per componenete temporale
K_time = sigma_time**2 * np.exp(-d_time**2 / (2 * length_scale_time**2))

# Matern kernel per componenete temporale
# a_time = 1/5.0
# K_time = sigma_time**2 * a_space**3 * np.exp(-a_time*d_time) * \
#    (a_time**2/3*d_time**2+a_time*d_time+1)

K_time += 1e-6 * np.eye(n_t)


# === Cholesky decomposition ===
# x = m +Lu dove m e' la media, L la dec di Cholesky del kernel mentre u e' una Gaussiana 0,I
# Nota che in questo caso x, definito come Z sotto, e' proprio estratto in questo modo,
# non c'e correlazione nelle entrate delle colonne (ma e' giusto cosi')
L_time = cholesky(K_time, lower=True)
L_space = cholesky(K_space, lower=True)

# === Sample from standard normal and transform ===
# Genero una amtrice Gaussina di shape (n_t, n_x*n_y) con ciascuna entratta estratta da una Gaussina 0,1
Z = rng.standard_normal(size=(n_t, n_x * n_y))  # shape (n_t, n_x*n_y)

# Apply temporal correlation: L_time @ Z
Z_time_corr = L_time @ Z  # shape (n_t, n_x*n_y)

# Apply spatial correlation: multiply each time step by L_space
data = np.array([L_space @ Z_time_corr[t, :]
                 for t in range(n_t)])  # shape (n_t, n_x*n_y)

# Reshape to final 3D tensor
data = data.reshape(n_t, n_x, n_y)


# Calcolo il variogramma in direzione della retta y=x, perche' e' quello che mi serve
h_estimation = np.array([])
M_variogram = np.array([])

# DATASET PER CALCOLARE IL VARIOGRAMMA
dataV = data[0, :, :]
# print(dataV)

# for per creare i diversi lag
for i in range(1, len(y_coords), 1):
    h_estimation = np.append(h_estimation,
                             np.linalg.norm([x_coords[i]-x_coords[0], y_coords[0]-y_coords[i]]))

# for per calcolare il variogramma di Matheron per qui lag
# Posso usare entrambe le direzioni perche ho istropia
# altrimenti usare solo una delle due direzioni
for i in range(len(h_estimation)):
    val = []
    # print('------------------SW -> NE---------------------')
    # print('h_i=', i)
    # Uso i valori del variogramma in direzione SW -> NE
    for row in range(i+1, len(h_estimation)+1, 1):
        # print('row=', row)
        for col in range(0, len(h_estimation)-i, 1):
            # print('col=', col)
            val = np.append(val, (dataV[row, col]-dataV[row-i-1, col+i+1])**2)
            # print('data 1 =', dataV[row, col])
            # print('data 2 =', dataV[row-i-1, col+i+1])
            # print('val=', val)
    # print('------------------SE -> NW---------------------')
    # Uso i valori del variogramma in direzione SE -> NW
    for row in range(0, len(h_estimation)-i, 1):
        # print('row=', row)
        for col in range(0, len(h_estimation)-i, 1):
            # print('col=', col)
            val = np.append(val, (dataV[row, col]-dataV[row+i+1, col+i+1])**2)
            # print('data 3 =', dataV[row, col])
            # print('data 4 =', dataV[row+i+1, col+i+1])
            # print('val=', val)
    # print('val=', val)
    n = len(val)
    M_variogram = np.append(M_variogram, 1/n*np.sum(val))
    # print('M_variogram=', M_variogram)


# MODELLO TEORICO STOU
# theta = [c, Var, a, lambd]
def modello_teoricoSTOU(h, theta):
    return 2*np.pi*theta[1]*theta[0]**2 - theta[2] * np.exp(-theta[3]*h)


# MODELLO TEORICO MSTOU
# ATTENZIONE: CAMBIARE RIGA 286, 291-296, 304, 448 (lambda_) e 615 (grafo variogramma)
# - objective function (che sia ols o wls1 o wls2)
# - initial guess + bounds
# - popt
# - esponente popt.x[3] in riga 463
# - grafici al fondo
# theta = [c, Var, b, alpha, beta]
def modello_teoricoMSTOU(h, theta):
    return (np.pi*theta[1]*theta[0]**2*gamma(theta[3]-3)*theta[4]**theta[3])/(4*gamma(theta[3])) - theta[2] * h**((3-theta[3])/2)


# OTTIMIZZAZIONE
# y=[Z_ti(xi) - Z_tj(xj)] dove ti(xi) e tj(xj) hanno distanza h
def ols_objective(theta, h, y):
    residuals = y - modello_teoricoSTOU(h, theta)
    return np.sum(residuals**2)


def wls1_objective(theta, h, y):
    W = np.array([2*i**2 for i in reversed(range(1, len(h)+1, 1))])
    residuals = W*(y - modello_teoricoSTOU(h, theta))
    return np.sum(residuals**2)


def wls2_objective(theta, h, y):
    W = np.array([2*i**2 for i in reversed(range(1, len(h)+1, 1))]
                 )/(np.mean(h)**2)
    residuals = W*(y - modello_teoricoSTOU(h, theta))
    return np.sum(residuals**2)


# Initial guess for parameters
initial_guess = np.array([1.0, 2.0, 1.0, 1.0])
bounds_STOU = [(0, None)] * len(initial_guess)  # Tutti i parametri > 0
# initial_guess = np.array([1.0, 2.0, 1.0, 4.0, 1.0])
# bounds_MSTOU = [(0, None), (0, None), (0, None),
#                (3, None), (0, None)]  # theta[3] > 3


# Perform the minimization
'''
popt = minimize(ols_objective,  initial_guess,
                args=(h_estimation, M_variogram), bounds=bounds_STOU)
'''
popt = basinhopping(ols_objective, initial_guess, minimizer_kwargs={
    "args": (h_estimation, M_variogram), "bounds": bounds_STOU}, seed=2)


print('estimated parameters:', popt.x)
data = tf.cast(data, tf.float64)
# ------------------------------------------------------------------------------------------------------------
# PRENDO DATI TEMPERATURE
# Apri il file (sostituisci 'tuo_file.nc' con il nome del tuo file)
'''
file_path = r"P:\Desktop\python\Embedding\2m_temperature_2018_1.40625deg.nc"
dataset = xr.open_mfdataset(file_path, mode="r")  # "r" = solo lettura

# Stampa le variabili presenti nel file
# print(dataset.variables.keys())

lat = dataset.variables.get('lat')[61:68]
long = dataset.variables.get('lon')[0:7]
# time = dataset.variables.get('time')
# temperatura superficiale terrestre
data = dataset.variables.get('t2m')[:, 61:68, 0:7]  # .data.ravel()

# DISTANZE A CUI CALCOLARE IL VAIOGRAMMA
x, y = np.meshgrid(lat, long, indexing='ij')
coordinates = np.stack([x.ravel(), y.ravel()], axis=-1)

h_estimation = np.array([])
M_variogram = np.array([])

# DATASET PER CALCOLARE IL VARIOGRAMMA
dataV = data[0, :, :]

# for per creare i diversi lag
for i in range(1, len(long), 1):
    h_estimation = np.append(h_estimation,
                             np.linalg.norm([lat[i]-lat[0], long[0]-long[i]]))

# for per calcolare il variogramma di Matheron per qui lag
for i in range(len(h_estimation)):
    val = []
    n = 0

    for r in range(i+1, len(h_estimation)+1, 1):
        for c in range(0, len(h_estimation)+1-i, 1):
            val = np.append(val, (dataV[r, c]-dataV[r-i, c+i])**2)
            n = n+1
    M_variogram = np.append(M_variogram, 1/n*np.sum(val))


# MODELLO TEORICO
# theta = [c, Var, a, lambd]


def modello_teorico(h, theta):
    return 2*np.pi*theta[1]*theta[0]**2 - theta[2] * np.exp(-theta[3]*h)

# OTTIMIZZAZIONE PER DATASET VERO
# y=[Z_ti(xi) - Z_tj(xj)] dove ti(xi) e tj(xj) hanno distanza h


def ols_objective(theta, h, y):
    residuals = y - modello_teorico(h, theta)
    return np.sum(residuals**2)


# Initial guess for parameters
initial_guess = np.array([1.0, 1.0, 1.0, 1.0])


# Perform the minimization
popt = minimize(ols_objective, initial_guess,
                args=(h_estimation, M_variogram))

data = data[range(23, 8759, 168), :, :]
data = tf.cast(data, tf.float64)
'''
# ------------------------------------------------------------------------------------------------------------

# Inizio del codice per creare previsione 1 time ahead
data_path = "datasets/"
normalize_data = False
use_different_eps = False


h_t = 0.05


center_pixel = 100


# TURN ON THE RIGHT DATASET!!


files = ["absB/absB1",  # CHANGE!!!!
         "sqB/sqB1",
         "absB/absB2",
         "sqB/sqB2",
         "absB/absB3",
         "sqB/sqB3"
         ]

directory = [
    "absB/",
    "sqB/",
    "absB/",
    "sqB/",
    "absB/",
    "sqB/"
]  # CHANGE BACK THE UNBOUNDED CHOICE


labels = ['absB']  # , 'sqB', 'absB', 'sqB', 'absB', 'sqB']  # 'absB','sqB']#

eps = [2.99]  # eps=[3,3]

ids = [0, 1, 2, 3]  # ,1,2,3]#,2]#,1,2,3]

labs = [0]  # ,1,3,5]
flags = [0]
path = 'reviewGraphics/fig4bis/'

colors = ['orange', 'crimson']


for j in labs:  # range(len(labels)):
    if not os.path.exists(path+directory[j]):
        os.makedirs(path+directory[j])

    # hf = h5py.File(path+files[j]+'.h5', 'w')

    for el in p:
        for current_id in ids:  # range(datanum):
            max_val = int(data.shape[0]-h)  #
            data_val = data[max_val:, :, :]
            data = data[:max_val, :, :]  # -2000

            # x=hf.create_group(str(datasetsON[current_id])+str(el))

        ground_truth = data_val[h-1, :, :]

        N, x_size, y_size = data.shape
        c = 1
        # initialize model to estimate parameters, which are independent of the position and
        # use set A, c, lambda to previously estimated values
        Z = STOU(tf.constant([[3, 3]]),
                 data, lambda_=popt.x[3], c_=popt.x[0], h_t=0.05)

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
        for flag in flags:
            data_current = data[:, -el:]  # form pth last column to the end

            # print("p: ", el)
            # print("p: ",el,file=outfile)
            cont = 0
            Z.N = N
            Z.p = el
            Z.a = a
            Z.k = 1
            # Z.set_model_for_benchmark(el)
            print('computing benchmark')
            # benchmarkA=bench(Z,data,x_size,el,c)

            [output, omin, omax, draws] = predict_h_slice(
                Z, data, h, x_size, y_size, labels[j], el, c=popt.x[0], Ndraws=50, eps=eps[j], mean=0, var=1)
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

            '''
            # a IS STORED WRT. FLAGS
            x = hf.create_group(str(datasetsON[current_id])+'p'+str(el)+'a'+str(flag))
            x.create_dataset('a', data=Z.a)
            x.create_dataset('draws', data=output)
            x.create_dataset('min', data=outmin[flag])
            x.create_dataset('max', data=outmax[flag])
            x.create_dataset('q25', data=out_25[flag])
            x.create_dataset('q75', data=out_75[flag])
            x.create_dataset('q025', data=out_025[flag])
            x.create_dataset('q975', data=out_975[flag])
            x.create_dataset('benchmark', data=benchmarkA[flag])
            x.create_dataset('test', data=ground_truth[el:-el])

            hf.close()
            '''
    # plt.figure()
    # axes.set_xlim([xmin,xmax])
    # plt.ylim((-2,2))

    # plt.fill_between(np.arange(data[el:-el,:].shape[0]),out_025.flatten(),out_975.flatten(),alpha=0.75,color='wheat')

'''
			plt.fill_between(np.arange(el,data[el:-el,:].shape[0]+el),outmin[0].flatten(),outmax[0].flatten(),alpha=0.55,color='silver')
			plt.fill_between(np.arange(el,data[el:-el,:].shape[0]+el),outmin[1].flatten(),outmax[1].flatten(),alpha=0.55,color='gainsboro')
			for flag in flags:
				plt.fill_between(np.arange(el,data[el:-el,:].shape[0]+el),out_25[flag].flatten(),out_75[flag].flatten(),alpha=0.75,color=colors[flag])	
					
			plt.plot(np.arange(el,ground_truth[el:-el].shape[0]+el),ground_truth[el:-el],linewidth=.8,color='black')
			#plt.plot(outmean,linewidth=.8,color='purple')
			plt.plot(np.arange(el,benchmarkA[el:-el].shape[0]+el),benchmarkA[el:-el],linewidth=.8,color='purple')
			#plt.plot(benchmarkS,linewidth=.8,color='green')			
				
			#plt.legend(["50-prediction range","quantiles range (.25-.75)","test set","linear model"])
			plt.savefig(path+directory[j]+'png/'+'FCfig4bis_' + datasetsON[current_id]+'_p'+str(el)+".png")#+'_'+str(datetime.now().strftime('%d_%H_%M'))
			plt.savefig(path+directory[j]+'pdf/'+'FCfig4bis_' + datasetsON[current_id] +'_p'+str(el)+".pdf")#++ '_'str(datetime.now().strftime('%d_%H_%M'))
			plt.savefig(path+directory[j]+'eps/'+'FCfig4bis_'  + datasetsON[current_id] +'_p'+str(el)+".eps",format='eps')#+'_'+str(datetime.now().strftime('%d_%H_%M'))
				
			plt.close()

'''

# print(np.array(omin).shape)
# plt.figure()

# print(data)
# print(np.mean(output, axis=2))

'''
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf1 = ax.plot_surface(x, y, ground_truth, color='Blue',
                        linewidth=0, antialiased=False)
surf2 = ax.plot_surface(x, y, np.array(omin), color='Green',
                        linewidth=0, antialiased=False)
surf3 = ax.plot_surface(x, y, np.array(omax), color='Green',
                        linewidth=0, antialiased=False)
surf4 = ax.plot_surface(x, y, np.array(np.mean(output, axis=2)), color="Red",
                        linewidth=0, antialiased=False)

surf5 = ax2.plot_surface(x, y, ground_truth, alpha=0.3, color='Blue',
                         linewidth=0, antialiased=False)
surf6 = ax2.plot_surface(x, y, np.array(np.mean(output, axis=2)), color="Red", alpha=0.7,
                         linewidth=0, antialiased=False)
'''

pixels = []
values = []
values_min = []
values_max = []
values_truth = []
values_025 = []
values_05 = []
values_25 = []
values_75 = []
values_95 = []
values_975 = []
for i in range(len(x[:, 0])):
    for j in range(len(y[0, :])):
        if (i < popt.x[0]*Z.p or j < popt.x[0]*Z.p) or (i > x_size-1-popt.x[0]*Z.p or j > y_size-1-popt.x[0]*Z.p):
            continue
        pixels.append(x[j, i])
        values.append(outmean[i, j])
        values_min.append(np.array(omin)[i, j])
        values_max.append(np.array(omax)[i, j])
        values_025.append(out_025[0][i, j])
        values_05.append(out_05[0][i, j])
        values_25.append(out_25[0][i, j])
        values_75.append(out_75[0][i, j])
        values_95.append(out_95[0][i, j])
        values_975.append(out_975[0][i, j])
        values_truth.append(ground_truth[i, j])


fig, ax3 = plt.subplots()

ax3.plot(range(len(values)), values, color='Purple')  # , marker='o')
ax3.plot(range(len(values)), values_min, color='Teal')  # , marker='x')
ax3.plot(range(len(values)), values_max, color='Teal')  # , marker='x')
ax3.plot(range(len(values)), values_truth, color='Black')  # , marker='s')

# ax3.fill_between(range(len(values)), values_025,
#                 values_975, color='0.5', alpha=0.2)
# ax3.fill_between(range(len(values)), values_05,
#                 values_95, color='orange', alpha=0.3)
ax3.fill_between(range(len(values)), values_min,
                 values_max, color='Teal', alpha=0.4)
ax3.fill_between(range(len(values)), values_25,
                 values_75, color='Crimson', alpha=0.4)

# ax3.plot(range(len(values)), values_025, color='0.5')
# ax3.plot(range(len(values)), values_05, color='orange')
ax3.plot(range(len(values)), values_25, color='Crimson')  # , marker='x')
ax3.plot(range(len(values)), values_75, color='Crimson')  # , marker='x')
# ax3.plot(range(len(values)), values_95, color='orange')
# ax3.plot(range(len(values)), values_975, color='0.5')

ax3.set_xlabel("Indice")
ax3.set_ylabel("Valore")
ax3.set_title('3d Plot')
plt.show()


fig, ax4 = plt.subplots()
ax4.plot(h_estimation, M_variogram, color='Red', marker='o')
# ax4.plot(h_estimation, modello_teoricoSTOU(
#    h_estimation, popt_minimize.x), color='Blue', marker='x')
ax4.plot(h_estimation, modello_teoricoSTOU(h_estimation,
                                           popt.x), color='Green', marker='x')

P = (len(pixels)**2)
averRMAE = np.sum(np.abs(tf.subtract(values_truth, values) / values_truth))/P
print('averRMAE=', averRMAE)


# surf=ax.plot_wireframe(lat,z,oz,rstride=10,cstride=10)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# \ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf1, shrink=0.5, aspect=5)

# plt.show()

# Show Figure
# plt.plot_surface()
