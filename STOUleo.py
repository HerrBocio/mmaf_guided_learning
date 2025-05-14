import os

import numpy as np
import tensorflow as tf
import scipy.linalg
from scipy.optimize import minimize

from samplerShift import get_gaussian_sampler, Sampler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_loss_function(A, b, loss, eps=2.99):
    '''
    Generates the loss function L(h(A),b) from a specific functios set.

    Parameters
    ----------
    A : tensorflow array
      Input vector from the training dataset.

    b : tensorflow array
      Output vector from the training dataset.

    loss : {'absUnB', 'sqUnB', 'absB', 'sqB'}
      Specifies the choice for the loss function.

    eps : float, optional
      Truncation value for the loss function. The default is 2.99.


    Returns
    -------
    r_eps : function
      Loss function.
    '''
    def fun_sq(beta): return (tf.einsum("nm,bm->bn", A,
                                        # sq LOSS!
                                        beta[:, :-1]) + tf.expand_dims(beta[:, -1], 1) - b)**2

    def fun_abs(beta): return np.abs(tf.einsum("nm,bm->bn", A,
                                               # abs LOSS!
                                               beta[:, :-1]) + tf.expand_dims(beta[:, -1], 1) - b)

    def fun_b(x): return (tf.cast(x <= eps, dtype=tf.float64) * x +
                          # eps-bounded function
                          tf.cast(x > eps, dtype=tf.float64) * eps)

    if loss.find('absUnB')+1:
        def r_eps(beta): return tf.reduce_mean(
            fun_abs(beta), axis=1)  # *m#ABS UNBOUNDED!

    elif loss.find('sqUnB')+1:
        def r_eps(beta): return tf.reduce_mean(
            fun_sq(beta), axis=1)  # *m#SQUARE UNBOUNDED!

    elif loss.find('absB')+1:
        def r_eps(beta): return tf.reduce_mean(
            fun_b(fun_abs(beta)), axis=1)  # *m#ABS BOUNDED!

    elif loss.find('sqB')+1:
        def r_eps(beta): return tf.reduce_mean(
            fun_b(fun_sq(beta)), axis=1)  # *m#SQUARE BOUNDED!

    return r_eps


def get_posterior_distribution(A, b, loss, m, eps=2.99, l=tf.constant(2.0, dtype=tf.float64), num_realizations=10000, mean=0.0, var=1.0):
    '''
    Construct an instance of the class for the posterio distribution

    Parameters
    ----------
    A : tensorflow array
      Input vector from the training dataset.

    b : tensorflow array
      Output vector from the training dataset.

    loss : {'absUnB', 'sqUnB', 'absB', 'sqB'}
      Specifies the choice for the loss function.

    eps : float, optional
      Truncation value for the loss function. The default is 2.99.

    l : tensorflow consant, optional
      Normalized constant. The default is tf.constant(2.0, dtype=tf.float64).

    num_realizations : integer, optional
      DESCRIPTION. The default is 10000.

    mean : float, optional
      Mean of the prior distribution. The default is 0.0.

    var : float, optional
      Variance of the prior distribution. The default is 1.0.


    Returns
    -------
    Sampler(prior, g_normalized, 1.0/c)
      Represents a sample for the posterio distribution.
    '''
    r_eps = get_loss_function(A, b, loss, eps=eps)  # ,m PUT M BACK!!!!
    # print(A.shape[1])
    prior = get_gaussian_sampler(A.shape[1]+1, mean=mean, var=var)

    def g_unormalized(beta):
        return tf.math.exp(-tf.sqrt(l)*r_eps(beta))  # *m

    realization = prior(num_realizations)
    c = tf.reduce_mean(g_unormalized(realization))

    def g_normalized(beta):
        return g_unormalized(beta)/c

    return Sampler(prior, g_normalized, 1.0/c)


def fit(X, params):
    return X.dot(params)


def cost_f(params, X, y):
    return np.sum(np.abs(y - fit(X, params)))


class STOU:
    # variogram left out: needed?
    def __init__(self, x_position, data, lambda_, c_, num_realizations=2000, h_t=0.05):
        """
        Initializes STOU model
        :param x_position: coordinate to be investigated
        :param data: spatial-temporal data
        :param num_realizations:
        :param max_variogram_value: maximum value for which the variogram will be determined
        """
        # self.estimate_parameters(data,max_variogram_value=max_variogram_value)
        data = tf.constant(data, dtype=tf.float64)

        self.x_position = tf.constant(x_position, shape=(1, 2), dtype=tf.int32)

        self.data = data

        self.num_realizations = num_realizations

        self.h_t = h_t

        self.c_ = c_
        self.lambda_ = lambda_

    def get_cone_shift(self):
        """
        Funzione che genera l'attributo
        :self.cone_shift
        serve per prendere le coordinate spzio temporali
        che ci sono dentro un cono
        """
        coord = []
        # t in [p-1, p-2, ..., 0]
        for t in reversed(range(self.p)):  # self.p = p+h-1
            # aggiunge alla fine di coord il vettore dentro tf.constant()
            coord.append(tf.constant([[- (t+1), v, u] for v in range(-int(np.floor(self.c_*(t + 1)/np.sqrt(2))+1), int(np.floor(self.c_*(t + 1)/np.sqrt(2))+1) + 1)
                         for u in range(-int(np.floor(self.c_*(t + 1)/np.sqrt(2))+1), int(np.floor(self.c_*(t + 1)/np.sqrt(2))+1) + 1)]))
            # dentro tf.cponstant() costruisco la matrice di righe v in range(..) e colonne -(t+1)
            # Dubbio: dato che sto prendendo i valori con indici spaziali tra le rette
            # x = -(t+1) e x = t+1, non starei supponendo c = 1 ?

        # prima unisco tutto sulle righe, esempio: [1, 2], [3, 4] -> [1, 2, 3, 4]
        coord = tf.concat(coord, axis=0)
        # doppia espansione sulle colonne. Dubbio: e' veramente encessario ??
        coord = tf.expand_dims(tf.expand_dims(coord, 0), 0)
        # print('get_cone shift', coord.shape)
        self.cone_shift = coord

    def set_p_parameter(self, p):
        """
        Funzione che genera l'attributo self.p e chiama la funzione get_cone_shift(),
        che crea l'attributo self.cone_shift
        """
        # print(p)
        self.p = p
        self.get_cone_shift()
        # self.estimate_remaining_parameters()

    # @tf.function
    def get_cone_(self, x, t):
        """
        Funzione che preleva i dati da self.data agli indici spazio-temporali [x,t]
        """
        cone_end_coordinates = tf.constant([x, t], dtype="int32")

        # altra espansione di coord
        cone_ends_coordinates = tf.expand_dims(cone_end_coordinates, 0)
        # Dubbio: che senso ha espandere due volte al fondo di get_cone_shift(self)
        # se poi qui sotto prendo solo le ultime due dimensioni ?
        cone_coordinates = cone_ends_coordinates+self.cone_shift[0, 0, :, :]
        # ha senso espandere cone_end_coordinates perche' devo usarlo per estrare da self.data, che e'
        # spatio-temporale (2d), le coordinate rappresentate da cone_coordinates
        A = tf.gather_nd(self.data, cone_coordinates)
        return A

    def get_cone(self, x):
        """
        Funzione che ti restituisce il raster data-cubes,
        ovvero il dataset filtrato con l'embedding
        x = spatial position
        """
        # x_size, N = data.shape
        # print("p: ",self.p, "\ta: ", self.a)
        # considero le punte dei coni, occhio che self.N e' escluso
        # print(self.p)
        cone_ends = tf.range(self.N % self.a + self.p, self.N, self.a)
        # print(f"1) cone_ends = {cone_ends}")

        cone_ends_shape = (self.N - 1 - self.N % self.a) / (self.a)
        # print("2) cone ends shape: ",cone_ends_shape)

        # cast a numero intero e non a tensore
        cone_ends_shape = tf.cast(
            tf.floor(cone_ends_shape) + 1, dtype=tf.int32)
        # print("3) cone ends shape: ", cone_ends_shape)

        # MEMORIA MASSACRATA
        # print(x.shape)
        x1 = x[:, 0]
        x1 = tf.expand_dims(tf.expand_dims(x1, 1), 1)
        # print(f"4) x1 after double expand_dims: {x1}")
        # print(f"5) dim x1 after double expand_dims: {x1.shape}")
        cone_ends = tf.expand_dims(tf.expand_dims(cone_ends, 1), 0)
        # print(f"6) cone_ends after double expand_dims: {cone_ends}")
        # sulla prima e terza colonna questo comand tf.broadcast_to() non fa nulla
        # print(f"7) dim cone_ends after double expand_dims: {cone_ends.shape}")
        # mentre la seconda, che sarebbero le righe dele matrici, la adatta a quella di cone_ends_shape
        x1 = tf.broadcast_to(x1, [x1.shape[0], cone_ends_shape, 1])
        # print(f"8) x1 after after broadcast with cone_edns_shape: {x1}")
        # dal momento che voglio costruire un elemento con x e cone_ends faccio il
        # broadcast di cone_ends per ottenere le stesse dimensioni di x
        cone_ends = tf.broadcast_to(cone_ends, x1.shape)
        # print(f"9) cone_ends after broadcast with x1.shape: {cone_ends}")
        # unisco le righe di x con quelle di cone_ends
        # esempio:
        # x = tf.constant([[[4, 4, 4, 4], [3, 3, 3, 3]], [[4, 4, 4, 4], [3, 3, 3, 3]]])
        # y =  tf.constant([[[1, 1, 1, 1], [2, 2, 2, 2]], [[1, 1, 1, 1], [2, 2, 2, 2]]])
        # z = tf.concat([x,y], axis=2)
        # return: [[[4 4 4 4 1 1 1 1],[3 3 3 3 2 2 2 2]], [[4 4 4 4 1 1 1 1],[3 3 3 3 2 2 2 2]]]
        cone_ends_coordinates = tf.concat((cone_ends, x1), axis=2)

        aux = np.tile(x[:, 1:], cone_ends_shape)
        aux = tf.expand_dims(aux, -1)

        # print(aux)
        # print(f"9-bis) cone_ends_coordinates = {cone_ends_coordinates}")

        cone_ends_coordinates = tf.concat(
            [cone_ends_coordinates, aux], axis=-1)
        # print(f"10) cone_ends_coordinates = {cone_ends_coordinates}")
        # print(f"11) cone_ends_coordinates with shape: {cone_ends_coordinates.shape}")

        b = tf.gather_nd(self.data, cone_ends_coordinates)

        cone_ends_coordinates = tf.expand_dims(cone_ends_coordinates, 2)
        # print(f"12) cone_ends_coordinates = {cone_ends_coordinates}")
        # print(f"13) cone_ends_coordinates shape = {cone_ends_coordinates.shape}")
        # print(f"14) cone_shifts shape = {self.cone_shift.shape}")

        # aggiubgiamo le colonne che abbiamo tolto ad x all'inizio

        cone_coordinates = cone_ends_coordinates + self.cone_shift
        # print(f"15) cone_coordinates = {cone_coordinates}")
        # print(f"16) cone_coordinates shape = {cone_coordinates.shape}")

        # print("cone_coord", cone_coordinates.shape,"\tdata",self.data.shape)
        A = tf.gather_nd(self.data, cone_coordinates)
        # print("out of cone gather")
        return A, b

    def set_model_for_benchmark(self, p):
        """
        Funzione che setta gli attributi
        della classe per confrontarsi con
        il modello benchmark
        """
        # time_start = time.perf_counter()
        self.set_p_parameter(p)

        A, b = self.get_cone(self.x_position)
        self.A = A[0, :, :]
        self.b = b[0, :]
        # print(self.A.shape,self.b.shape)
        design_matrix = np.concatenate(
            (self.A.numpy(), np.ones((self.A.shape[0], 1))), axis=1)
        mat = scipy.linalg.pinv(
            np.matmul(np.transpose(design_matrix), design_matrix))
        mat = np.matmul(mat, np.transpose(design_matrix))
        self.beta_linear_modelS = np.matmul(mat, self.b)
        # print(self.beta_linear_modelS.shape)
        self.beta_linear_modelA = minimize(cost_f, np.ones(
            (self.beta_linear_modelS.shape[0],), dtype=np.float64), args=(design_matrix, b)).x
        # print(self.beta_linear_model)

    def set_model(self, p):
        """
        Funzione che setta gli attributi
        della classe del nostro embedding
        """
        self.set_p_parameter(p)

        A, b = self.get_cone(self.x_position)
        self.A = A[0, :, :]
        self.b = b[0, :]
        self.m = self.A.shape[0]

    def set_model_for_p_value(self, p, c, loss, eps=2.99, mean=0.0, var=1):
        """
        Funzione che setta gli attributi
        della classe per confrontare il p-value
        Divisa in due in set_model() +
        """
        self.set_p_parameter(p)

        # print(self.x_position)
        A, b = self.get_cone(self.x_position)
        self.A = A[0, :, :]
        # print(A.shape)
        self.b = b[0, :]
        self.m = self.A.shape[0]
        self.l = tf.cast(tf.floor(self.m/self.k), dtype=tf.float64)
        self.r_eps = get_loss_function(self.A, self.b, loss, eps)
        # qui sotto definisco l'attributo self.sampler usando una funzione definita
        # che restituisce un elemento di classe Sampler
        self.sampler = get_posterior_distribution(
            self.A, self.b, loss, self.m, eps=eps, l=self.l, num_realizations=100000, mean=mean, var=var)

    def stouPar(self, tau, c):
        """
        Ignota: oltre a non capire che oggetto amtematico rappresenta,
        self.lambda_ da dove esce ?
        """
        R_Y = np.zeros((3, 3), dtype=np.float64)
        r_y = np.ones((1, 3), dtype=np.float64)*tau**2*c*tf.math.exp(-3 *
                                                                     self.lambda_)/c*self.lambda_*(1+1/(2*self.lambda_))

        for i in range(R_Y.shape[1]):
            R_Y[i, i] = 2*tau*2*c/(4*self.lambda_**2)

        aux = tau**2*tf.math.exp(-self.lambda_/c) / \
            (2*self.lambda_)  # just a factorized term

        R_Y[0, 1] = aux*(1+c/self.lambda_)
        R_Y[1, 0] = R_Y[0, 1]
        R_Y[1, 2] = R_Y[0, 1]
        R_Y[2, 1] = R_Y[0, 1]

        R_Y[0, 2] = aux*tf.math.exp(-self.lambda_/c)*(2+c/self.lambda_)
        R_Y[2, 0] = R_Y[0, 2]
        return R_Y, r_y

    def post_predictive(self, tau=1, c=1, mu=0):
        """
        Per d=1 puoi calcolarti la predictive distribution di STOU,
        utilissima per fare i grafi
        *Per d>1 sei ingulat
        """
        R_Y, r_y = self.stouPar(tau, c)

        coord = []
        pred_mean = []
        pred_var = []
        for t in reversed(range(self.p)):
            coord.append(tf.constant([[v, - (t + 1)]
                         for v in range(-c*(t + 1), c*(t + 1) + 1)]))
        coord = tf.concat(coord, axis=0)
        coord = tf.expand_dims(tf.expand_dims(coord, 0), 0)

        for x_coord in range(self.data.shape[0]):
            self.x_position = tf.constant(x_coord, shape=(1,), dtype=tf.int32)

            if x_coord < self.p or x_coord > self.data.shape[0]-1-self.p:
                pred_mean.append(0)
                pred_var.append(0)
                continue

            cone_end_coordinates = tf.constant(
                [x_coord, self.data.shape[1]], dtype="int32")
            # print("\tbefore ends: ", cone_end_coordinates)
            cone_ends_coordinates = tf.expand_dims(cone_end_coordinates, 0)
            # print("\t\tafter ends: ", cone_end_coordinates)
            cone_coordinates = cone_ends_coordinates + coord[0, 0, :, :]
            # print("cone: ", cone_coordinates)
            A = tf.gather_nd(self.data, cone_coordinates)
            # print(A.shape)
            mean = 2*c*mu*self.lambda_**2 + \
                np.matmul(np.matmul(r_y, scipy.linalg.pinv(R_Y)),
                          (np.transpose(A)-2*c*mu/self.lambda_**2))
            # print(np.mean(mean))
            pred_mean.append(np.mean(mean))
            var = c*tau**2/(2*self.lambda_**2)*(1-np.matmul(r_y,
                                                            np.matmul(scipy.linalg.pinv(R_Y), np.transpose(r_y))))
            # print(np.mean(var))
            pred_var.append(np.mean(var))
        return pred_mean, pred_var

    def sample_prior(self, batch_size, beta_shape):
        """
        Genera un campione estratto da una Gaussiana
        """
        out = tf.random.normal((batch_size, beta_shape), dtype=tf.float64)
        return out

    def infer_beta(self, num=100000):
        """
        Funzione che fa la media della Gibbs distribution
        """
        beta = self.sampler.prior_sampler(num)
        beta = tf.reduce_mean(tf.expand_dims(self.sampler.g(beta), 1)*beta, 0)
        # print("\t\tshape> ",beta.shape)
        return beta

    def infer_beta_non_averaged(self, num=1):
        """
        Funzione che fa il sampling della Gibbs
        vettore che restituisce estrazioni dalla Gibbs
        """
        beta = self.sampler.sample(num)

        # print("\tprevShapeNA> ", beta.shape)
        # beta=tf.reduce_mean(beta,0)
        # print("\t\tpostShapeNA> ",beta.shape)
        return beta

    def init_ffnn(self, n1, n2, N):
        """
        Funzione che serve per settare i pesi della rete neurale
        estraendoli dalla Gibbs distribution
        """
        beta = self.sampler.sample(N*n1*n2)
        # print("\tprevShapeNA> ",beta.shape)
        # beta=tf.reduce_mean(beta,0)
        # print("\t\tpostShapeNA> ",beta[:,-1].numpy().reshape(n1,n2).shape)
        return beta[:, -1].numpy().reshape(n1, n2, N)

    def estimate_a_k(self, p, eps, flag=5):  # ,flagG,flagN):
        """
        Questo serve per settare gli attributi
        self.a
        self.k
        utilita' ignota dal momento che self.k e'
        sempre settato = 1
        """
        # print("flagG",flagG,"flagN",flagN)
        k = 0
        if flag == 0:  # and flagG==0:
            # print("in est Gau")
            # print("in est Gau",file=outfile)
            found_value = False
            for a_ in range(p+1, self.N-2+1):
                # print(a_, self.helper1(a_))
                if self.helper1(a_) > 0:
                    a_final = a_
                    found_value = True
                    break
            # print(a_)
            self.a = a_
            self.k = 1  # K IS FIXED!!!!!

        if flag == 1:  # and flagN==0:
            found_value = False
            for a_ in range(p+1, self.N-2+1):
                # print(a_, self.helper2(a_))
                if self.helper2(a_) > 0:
                    a_final = a_
                    found_value = True
                    break
            # print(a_)
            # Dubbio: Forse self.a = a_final
            self.a = a_
            self.k = 1  # K IS FIXED!!!!!

    def helper1(self, x):
        # print(self.lambda_, self.h_t,self.p,self.N,x)
        return self.lambda_*self.h_t*pow(x, (3./2)) - self.lambda_*self.p*self.h_t*pow(x, (1./2)) - 3*np.sqrt(self.N)

    def helper2(self, x):
        return x+np.log(x/(2*self.N))/(self.lambda_*self.h_t) - self.p

    def helper_new_Nig(self, p, eps=2.99):
        h_t = self.h_t
        p_t = p*h_t
        # val=(self.lambda_*self.p-1)+np.sqrt((self.lambda_*self.p-1)**2.0+4*self.lambda_*h_t*(e+1)/e*self.N)

        # print("in Hnig")

        val = np.exp(2*eps/self.lambda_)/h_t + p

        # print("stuck nig?")

        k = 0
        a_final = 0
        found_value = False
        while True:
            k = k+1
            # print("new k: ",k)
            # print("new k: ",k,file=outfile)
            for a_ in range(p+1, min(800, self.N-2+1)):
                # print("\ta prop: ", a_)
                # print("\ta prop: ",a_,file=outfile)
                if a_*k >= val:
                    a_final = a_
                    found_value = True
                    break
            if found_value:
                break
        # for k in range

        return a_final, k

    def helper_new_Gau(self, p, eps=3):
        h_t = self.h_t
        p_t = p/h_t
        # val=(self.lambda_*self.p-1)+np.sqrt((self.lambda_*self.p-1)**2.0+4*self.lambda_*h_t*(e+1)/e*self.N)
        # val=(self.lambda_*self.p*h_t-1.0)+ np.sqrt( (self.lambda_*self.p*h_t-1.0)**2.0+8.0*h_t*self.lambda_*self.N )

        val = ((eps+1)/self.lambda_)/h_t + p
        k = 1
        a_final = np.floor(val)
        found_value = False
        while True:
            k = k+1
            # print("new k: ",k)
            # print("new k: ",k,file=outfile)
            for a_ in range(p+1, min(800, self.N-2+1)):
                # print("\ta prop: ", a_)
                # print("\ta prop: ",a_,file=outfile)
                if a_*k >= val:
                    a_final = a_
                    found_value = True
                    break
            if found_value:
                break
        # for k in range
        return a_final, k
