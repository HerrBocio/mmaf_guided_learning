from jax import grad,vmap
import jax.numpy as jnp
import jax.random


class MyMultiNormalDiagFromLogScale:	
#to be placed in a specific script (collect other distros?)
#rewrite class wrt to functional syntax
  """
  Class of multivariate normal distribution.
  Currently supports only distribution w/ diagonal covariance matrix  
  """

  def __init__(self, loc, nu,seed):
    '''
    Class constructor
    Inputs: 
          loc: mean
          nu: log scale
          seed: random PNRG key
    '''
    
    self._var = jnp.exp(nu)
    self._log_scale = nu /2
    self._mean = loc
    self._param_shape = self._mean.shape
    self.seed=seed
    

  def sample(self, sample_size):
    '''
    Method for sampling sample_size times from the distribution
    '''
    
    subkeys=jax.random.split(self.seed,num=sample_size)
    sample_shape = self._param_shape
    sam=vmap(lambda k : jax.random.normal(k, shape=sample_shape) * jnp.exp(self._log_scale) + (self._mean) )(subkeys)
    return sam

  def log_prob(self, x):
    '''
    Method for computing the log density of the distribution
    '''
    log_prob = jax.scipy.stats.multivariate_normal.logpdf(x,mean=self._mean, cov=jnp.diag(self._scale))
    sum_axis = [-(i + 1) for i in range(len(self._param_shape))]
    return jnp.sum(log_prob)
    

def my_multi_normal(key,*params,) :
  '''
  Function that instantiates the class MyMultiNormalDiagFromLogScale 
  '''
  return MyMultiNormalDiagFromLogScale(loc=params[0],nu=params[1],seed=key)#, scale=jnp.diag(params[1]))



def sge_pwj(score_function,params,dist_builder,rng,num_samples=1):
  '''
  Function that computes the pathwise gradient estimator for a generic distribution
  Input:
      score_function: score function
      params: parameters of the distribution
      dist_builder: function that calls the instantiation of the distribution class
      rng: random PRNG key
      num_samples: number of samples from the distribution
  Output:
      val: value of the expected score function estimator
      grad: stochastic gradient estimator
  '''
  def surrogate(params):
      # We vmap the function application over samples - this ensures that the
      # function we use does not have to be vectorized itself.
      dist = dist_builder(rng,*params)
      eu=vmap(score_function)(dist.sample((num_samples,)))
      return jnp.mean(eu)

  val_=surrogate(params)
  grad_=grad(surrogate)(params)
  return [val_,grad_]

def sge_pwj_2(function,params,dist_builder,rng,num_samples=1):
  '''
  Function that computes the score function gradient estimator for a generic distribution
  Input:
      score_function: score function
      params: parameters of the distribution
      dist_builder: function that calls the instantiation of the distribution class
      rng: random PRNG key
      num_samples: number of samples from the distribution
  '''   
  def surrogate(params):
      # We vmap the function application over samples - this ensures that the
      # function we use does not have to be vectorized itself.
      dist = dist_builder(rng,*params)
      return (vmap(function)(dist.sample((num_samples,))))


  
  return grad(surrogate)(params)

