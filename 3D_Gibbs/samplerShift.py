import tensorflow as tf
import numpy as np


def get_gaussian_sampler(dim, mean=0.0, var=1.0):
    '''
    Returns a generator for simulating dim-dimensional normally distributed random vectors.

    Parameters
    ----------
    dim : integer
      Size of the prameter vector B_1 of the linear predicotr B_0+B_1*X.

    mean: float, optional
      Mean of the prior distribution.

    var: float, optional
      Covariance matrix of the prior distribution.


    Returns
    -------
    out : function
      Function that generates a normally distributed random vector from its size.
    '''
    def fun(batch_size): return tf.random.normal((batch_size, dim), dtype=tf.float64,
                                                 mean=mean, stddev=np.sqrt(var))
    return fun


class Sampler:

    def __init__(self, prior_sampler, g, norm_cost):

        self.prior_sampler = prior_sampler
        self.g = g
        self.n_c = norm_cost
        self.dim = self.prior_sampler(1).shape[1]

    def sample(self, N):
        '''
        Returns N samples of length equal to the input size X from the posterior distribution via acceptance rejection.

        Parameters
        ----------
        N : integer
          Number of desired samples.


        Returns
        -------
        out : array
          Samples from the posterior distribution.
        '''
        out = tf.zeros((0, 1, self.dim), dtype=tf.float64)
        current_N = out.shape[0]

        while current_N < N:
            #print("samplig da gibbs:", current_N)
            realization = self.sample_step(N)
            out = tf.concat((out, realization), 0)
            current_N = out.shape[0]

        out = out[:N, 0, :]

        return out

    def sample_step(self, N=1000):
        '''
        Performs acceptance rejection.

        Parameters
        ----------
        N : integer, optional
          Number of samples from the prior. The default is 1000.


        Returns
        -------
        out : array
          Sample of variable size, consisting on the accepted draws from the posterior distribution.
        '''
        X = self.prior_sampler(N)
        U = tf.random.uniform((N,), dtype=tf.float64)
        pos = U <= self.g(X) / self.n_c
        out = tf.gather(X, tf.where(pos))

        return out
