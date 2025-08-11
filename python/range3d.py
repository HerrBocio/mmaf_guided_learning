# import h5py
from scipy.io import loadmat
# import matplotlib.pyplot as plt

# import pandas as pd
# import csv
# from datetime import datetime

from tqdm import trange

# import tensorflow_probability as tfp

import numpy as np
import tensorflow as tf

# from seaborn import heatmap,color_palette as col


# from STOUShift import STOU
# from samplerShift import get_gaussian_sampler, Sampler

# from drawRMSE import drawRMSE
# from averRMSE import averRMSE

# import time
# import os


def bench(Z, data, x_size, p, c):
    """
    """
    betas_linear = []
    for x_coord in range(data.shape[0]):
        # for x_coord in range(5):

        # print("id: ", current_id,":",x_coord,' of ', x_size)
        if x_coord < p or x_coord > x_size-1-p:
            betas_linear.append(np.nan)
            continue
        Z.x_position = tf.constant(x_coord, shape=(1,), dtype=tf.int32)
        Z.set_model_for_benchmark(p)
        # print("Linear:",Z.beta_linear_model,np.sqrt(np.sum(np.array(Z.beta_linear_model)**2.0)))
        betas_linear.append(Z.beta_linear_modelA)

    pred = predict_slice_old(data, p, c, betas_linear)
    return pred[p:-p]


def predict_slice_old(data, p, c, models):
    next_slice = np.zeros((data.shape[0], 1), dtype="float64")

    coord = []
    for t in reversed(range(p)):
        coord.append(tf.constant([[v, - (t + 1)]
                     for v in range(-c*(t + 1), c*(t + 1) + 1)]))
    coord = tf.concat(coord, axis=0)
    coord = tf.expand_dims(tf.expand_dims(coord, 0), 0)

    for i in range(len(models)):
        # if len(giessen_models[i])
        if not hasattr(models[i], "__len__"):
            next_slice[i] = np.nan
            continue
        if i < p:
            next_slice[i] = np.nan
            continue
        if i > data.shape[0]-1-p:
            next_slice[i] = np.nan
            continue
        cone_end_coordinates = tf.constant([i, data.shape[1]], dtype="int32")
        cone_ends_coordinates = tf.expand_dims(cone_end_coordinates, 0)
        cone_coordinates = cone_ends_coordinates + coord[0, 0, :, :]
        A = tf.gather_nd(data, cone_coordinates)

        if np.sum(np.isnan(A)) > 0:
            next_slice[i] = np.nan
            continue

        val = tf.reduce_sum(A*models[i][:-1])+models[i][-1]

        next_slice[i] = val.numpy()
    return next_slice


def predict_slice(Z, data, x_size, y_size, loss, p, c, Ndraws, eps, mean=0.0, var=1):
    """
    Funzione che ti predice il punto del cono piu' nuovo
    usando tutto il cono nuovo (tranne la punta in cui vogliamo fare la previsione)
    """

    # next_slice =np.zeros((data.shape[0] ,1) ,dtype="float64")

    # print("\n in pred_slice")

    # print("\n in pred_slice",file=outfile)
    output = np.array(np.zeros((data.shape[1], data.shape[2], Ndraws)))

    # print("out in pred: ",output.shape)

    coord = []
    # print(data.shape[0])
    outmin = np.array(np.ones((data.shape[1], data.shape[2]))*np.inf)
    outmax = np.array((-1.)*np.ones((data.shape[1], data.shape[2]))*np.inf)

    # t in [p-1, p-2, ..., 0]
    for t in reversed(range(p)):  # self.p = p
        # aggiunge alla fine di coord il vettore dentro tf.constant()
        coord.append(tf.constant([[- (t+1), v, u] for v in range(-int(np.floor(c*(t + 1)/np.sqrt(2))+1), int(np.floor(c*(t + 1)/np.sqrt(2))+1) + 1)
                     for u in range(-int(np.floor(c*(t + 1)/np.sqrt(2))+1), int(np.floor(c*(t + 1)/np.sqrt(2))+1) + 1)]))
        # dentro tf.cponstant() costruisco la matrice di righe v in range(..) e colonne -(t+1)
        # Dubbio: dato che sto prendendo i valori con indici spaziali tra le rette
        # x = -(t+1) e x = t+1, non starei supponendo c = 1 ?

    # prima unisco tutto sulle righe, esempio: [1, 2], [3, 4] -> [1, 2, 3, 4]
    coord = tf.concat(coord, axis=0)
    # doppia espansione sulle colonne. Dubbio: e' veramente encessario ??
    coord = tf.expand_dims(tf.expand_dims(coord, 0), 0)

    x_models = []
    draws = []
    # mu,sigma=Z.post_predictive()

    # print(el)
    # ◙AS = []

    for i in trange(x_size, desc='posterior', colour='red'):
        # making the Ndraws
        # if len(models[i]
        y_models = []
        for j in range(y_size):  # ,desc='posterior',colour='red'): #making the Ndraws

            # print("exceeds")
            # if it's outside the cone then skip
            if (i < c*p or j < c*p) or (i > x_size-1-c*p or j > y_size-1-c*p):
                # print("if true")
                # p_values.append(-1)
                # print("    p:",p_values[-1])
                y_models.append(np.nan)
                continue

            Z.x_position = tf.constant([[i, j]], shape=(1, 2), dtype=tf.int32)

            cone_end_coordinates = tf.constant(
                [data.shape[0], i, j], dtype="int32")
            # print("\tbefore ends: ", cone_end_coordinates)
            cone_ends_coordinates = tf.expand_dims(cone_end_coordinates, 0)
            # print("\t\tafter ends: ", cone_end_coordinates)
            cone_coordinates = cone_ends_coordinates + coord[0, 0, :, :]
            # print("cone: ", cone_coordinates)
            A = tf.gather_nd(data, cone_coordinates)
            # print(A.shape)
            # AS.append[tf.matmul(A,tf.ones((A.shape[0],A.shape[0])))]
            # print(AS[i].shape)
            # print("out of gather")
            # print("\nA: ",A.shape)
            # slices data wrt/ cone coordinate shape

            # ,mean=mu[i],var=sigma[i])
            Z.set_model_for_p_value(p, c, loss, eps, mean=mean, var=var)

            model = Z.infer_beta_non_averaged(num=Ndraws)
            # print("model shape", i, model.shape)
            y_models.append(model)

            # print("models shape",models[i][:].shape)
            # print("model appended: ", i )

            # print("draws shape v2: ",np.shape(draws[0]))

            # print(models.shape)
        x_models.append(y_models)

    # actual Ndraws predictions happen here
    # print(len(x_models))
    for r in trange(Ndraws, position=0, leave=True, desc="r", colour='green'):
        # print("r: ",r)
        # print("model ",r,": ", draws[r][80])

        for i in range(x_size):
            # if len(models[i])
            # print("pre gather")
            for j in range(y_size):
                # if it's outside the cone then skip
                if (i < c*p or j < c*p) or (i > x_size-1-c*p or j > y_size-1-c*p):
                    # print("if true")
                    # p_values.append(-1)
                    # print("    p:",p_values[-1])
                    # models.append(np.nan)
                    outmin[i][j] = np.nan
                    outmax[i][j] = np.nan
                    output[i][j][r] = np.nan
                    continue
                Z.x_position = tf.constant(
                    [[i, j]], shape=(1, 2), dtype=tf.int32)

                # MIGHT BE USEFUL TO STORE ALL A s somewhere

                # print("out of gather")
                # print("\nA: ",A)
                # slices data wrt/ cone coordinate shape

                if np.sum(np.isnan(A)) > 0:
                    outmin[i][j] = np.nan
                    outmax[i][j] = np.nan
                    output[i][j][r] = np.nan
                    # print("\n\t slice_%d: ",i,next_slice[i])

                    continue

                cone_end_coordinates = tf.constant(
                    [data.shape[0], i, j], dtype="int32")

                cone_ends_coordinates = tf.expand_dims(cone_end_coordinates, 0)
                cone_coordinates = cone_ends_coordinates + coord[0, 0, :, :]
                A = tf.gather_nd(data, cone_coordinates)
                '''
                if np.sum(np.isnan(A))>0:
                    next_slice[i][j]=np.nan
                    continue
                '''
                value = tf.reduce_sum(
                    A*x_models[i][j][r, :-1])+x_models[i][j][r, -1]

                # value=tf.reduce_sum(AS[i]*models[i][r,:-1]) + models[i][r,-1]

                # HERE GOES FFN

                output[i][j][r] = value

                if value <= outmin[i][j]:
                    # print("min changed: ",i, outmin[i],val)
                    outmin[i][j] = value

                    # print(outmin)
                if value >= outmax[i][j]:
                    # print("max changed: ",i, outmax[i],val)
                    outmax[i][j] = value
                    # print(outmax)

            # could add quantiles
        # draws.append(models)

        # prediction given by linear predictor h_beta(X) = beta_0 + beta^T X

        # print("\n\t slice_%d: ",i,next_slice[i])

    # output=output[p:-p][:]

    # print('\nlunghezza', len(outmin[0]))
    # print(outmin)
    return [output, outmin, outmax, draws]


def predict_all_possible_slices(Z, data_current, p, c, eps):
    # compute the actual predction with correctly gathered data, tuned parameters and
    # sample class (prior, posterior and normalizer)
    [out, _, _, _] = predict_slice(Z, data_current, p, c, Ndraws=1, eps=eps)
    data_current = np.concatenate((data_current[:, 1:], out), axis=1)

    # if last stage has all nan, skip
    while np.sum(np.isnan(out[:, -1])) < data_current.shape[0]:
        [next_slice, _, _, _] = predict_slice(
            Z, data_current, p, c, Ndraws=1, eps=eps)
        data_current = np.concatenate(
            (data_current[:, 1:], next_slice), axis=1)
        # print("\t dim data: ",data_current.shape)
        out = np.concatenate((out, next_slice), axis=1)
        # print("\t\t dim out: ", out.shape)
        # print((out))
        # print("in pred all possibile slices")
        # print(np.sum(np.isnan(out[:,-1]))<data_current.shape[0])
    out = out[:, :-1]
    return out


def get_simulated_data(filename):
    data = loadmat(filename+'.mat')
    data = data["data"]

    # data=data-np.mean(data)
    # data=data/np.std(data)

    return data


# mu,sigma=Z.post_predictive()
