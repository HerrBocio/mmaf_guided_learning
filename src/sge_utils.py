from jax import grad,vmap,jit
import jax.numpy as jnp
import jax.random
from src.utils import KLdiag_from_log_scale,chi2_diag_gaussians, dist_sample, Lip_realizations_masked
from functools import partial

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
    self._log_scale = nu/2
    self._mean = loc
    self._param_shape = self._mean.shape
    self.seed=seed
    

  def sample(self, sample_size):
    '''
    Method for sampling sample_size times from the distribution
    '''
    subkeys=jax.random.split(self.seed,num=sample_size)
    sample_shape = (sample_size,self._param_shape[0])
    sam=vmap(lambda k : jax.random.normal(k, shape=sample_shape) * jnp.exp(self._log_scale) + (self._mean) )(subkeys)
    return sam

  def log_prob(self, x):
    '''
    Method for computing the log density of the distribution
    '''
    log_prob = jax.scipy.stats.multivariate_normal.logpdf(x,mean=self._mean, cov=jnp.diag(self._scale))
    sum_axis = [-(i + 1) for i in range(len(self._param_shape))]
    return jnp.sum(log_prob)
    

def my_multi_normal(key,*params) :
  '''
  Function that instantiates the class MyMultiNormalDiagFromLogScale 
  '''
  return MyMultiNormalDiagFromLogScale(loc=params[0],nu=params[1],seed=key)#, scale=jnp.diag(params[1]))


#@partial(jit, static_argnums=(0,2))
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
  #print('ns',num_samples)
  def surrogate(params):
      # We vmap the function application over samples - this ensures that the
      # function we use does not have to be vectorized itself.
      dist = dist_builder(rng,*params)
      eu=vmap(score_function)(dist.sample(num_samples))
    
      return jnp.mean(eu)

  val_=surrogate(params)

  #traced=jax.jit(grad(surrogate)).trace(params)
  #lowered=traced.lower()
  #compiled=lowered.compile()
  #print('flops ER\n\t', compiled.cost_analysis()[0]['flops'])
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




#################################### PAC BOUNDS #########################################  
   

def truePAC(model,m,params):
    '''
    Function that computes the value of Pac bound (3.14) in Curato et al. (2025)
    Input:
        params: parameters of the generalized distribution
        piParams: parameters of the reference distribution
        eps: truncation level of the loss function
        Lip: Lipschitz constant estimate of the network, computed over the reference distribution
        delta: probability level of the pav bound
        a_p_c: dimension of the input 
        m: number of cones of the embedding
        alph: hyperparameter alpha
        p: cone lenght
        dim: size of the network
        arch: network structure
        chi: chi-square divergence between generalized posterior and reference distribution
    Output: 
        bb: pac bound
    '''
    bb=2*(-jnp.log(model.delta))/jnp.sqrt(m) + (.5*model.eps**2)/jnp.sqrt(m)
    kl=KLdiag_from_log_scale(model.pi_params,params)
    chi=chi2_diag_gaussians(model.pi_params,params)

  
    theta=(model.Lip*model.inp_size + 1 )
    bb+= (1./jnp.sqrt(m))*kl + jnp.sqrt(theta/(2*m)*chi)
    return bb


def pacBound(model,m,params):
    '''
    Function that computes the value of the linearised Pac bound (3.14) in Curato et al. (2025)
    Input:
        params: parameters of the generalized distribution
        piParams: parameters of the reference distribution
        eps: truncation level of the loss function
        Lip: Lipschitz constant estimate of the network, computed over the reference distribution
        delta: probability level of the pac bound
        a_p_c: dimension of the network input 
        m: number of cones of the embedding
        alph: hyperparameter alpha
        p: cone lenght
        dim: size of the network
        arch: network structure
        chi: chi-square divergence between generalized posterior and reference distribution
    Output: 
        bb: pac bound
    '''
    bb=0
    piParams=[model.pi_params[0],model.pi_params[1]]#jnp.diag(piParams[1])]
    bb= 2*(-jnp.log(model.delta))/jnp.sqrt(m) + (.5*model.eps**2)/jnp.sqrt(m)
    kl=KLdiag_from_log_scale(model.pi_params,params)
    theta=(model.Lip*model.inp_size+1)  #lambda gamma: 
    bb+= (1./jnp.sqrt(m))*kl+jnp.sqrt((theta/m)*kl*2)
    return bb




def pac_approx(model,mapped_params,m):
    
    '''
    Function that computes the linearised Pac bound (3.14) in Curato et al. (2025)
    Input:
        piParams: parameters of the reference distribution
        rhoParams: parameters of the generalised posterior distribution
        NNsize: parameter dimension
        m: number of cones of the embedding
        Lip: Lipschitz constant estimate of the network, computed over the reference distribution
        a_p_c: dimension of the network input     
    '''
    return KLdiag_from_log_scale(model.pi_params,mapped_params)*1./jnp.sqrt(m) + jnp.sqrt((model.Lip*model.inp_size+1)*2*KLdiag_from_log_scale(model.pi_params,mapped_params)/m)




def l_empirical_risk(model,batch):
  #lambda function for the computation of the empirical risk (to compute the gradient over)
  return lambda alpha: empirical_risk(model,batch,alpha)

def empirical_risk(model,batch,realization):
    '''
    single realization must have shape: (1,dim)
    '''
    masking= lambda alpha: model.pozzo(alpha)
    realization=vmap(masking,in_axes=0)(realization)
    empR=lambda beta : model.ffnn_loss_forward_pass(batch[:-1,:],beta,batch[-1,:],model.eps)
    eU=vmap(empR)(realization)
    return jnp.mean(eU,axis=0)


#################################### UNBOUNDED #########################################  

def hX_computation(model,batch,realization):
    '''
    single realization must have shape: (1,dim)
    '''
    masking= lambda alpha: model.pozzo(alpha)
    realization=vmap(masking,in_axes=0)(realization)
    empR=lambda beta : model.for_hX_comp(batch[:-1,:],beta)
    eU=vmap(empR)(realization)
    return eU

def target_function_unbounded_KL(piParams,rhoParams,m):
   return KLdiag_from_log_scale(piParams,rhoParams)/jnp.sqrt(m)

def target_function_unbounded_without_empRisk(model, batch, theta, VarZtrx, realization):
  """
  single realization must have shape: (1,dim)
  """
  m = batch[-1].shape[0]
  masking= lambda alpha: model.pozzo(alpha)
  hX=hX_computation(model,batch,realization)
  weights=vmap(masking,in_axes=0)(realization)
  Liphs = Lip_realizations_masked(weights)
  rho_Liph = jnp.mean(Liphs)
  rho_Liph_sq = jnp.mean(jnp.power(Liphs,2))
  abs_mean = lambda beta: jnp.abs(jnp.mean(beta))
  abs_E_hX = jax.vmap(abs_mean)(hX)
  rho_abs_E_hX_Liph = jnp.mean(jnp.multiply(abs_E_hX, Liphs))
  rho_hXsq = jnp.mean(jnp.power(abs_E_hX,2))
  tf = model.inp_size*rho_Liph*(theta + VarZtrx/jnp.sqrt(m))
  tf += rho_hXsq/(2*jnp.sqrt(m))
  tf += model.inp_size*theta/model.delta * rho_Liph
  tf += model.inp_size * theta / jnp.sqrt(m) * rho_abs_E_hX_Liph
  tf += model.inp_size**2*rho_Liph_sq/(2*jnp.sqrt(m))*(VarZtrx+theta**2)
  return tf

def target_function_unbounded(model, batch, theta, VarZtrx, realization):
  """
  single realization must have shape: (1,dim)
  """
  tf = empirical_risk(model, batch, realization)
  tf += target_function_unbounded_without_empRisk(model, batch, theta, VarZtrx, realization)
  return tf

def tf_unbounded(model, batch, theta, VarZtrx):
   return lambda beta: target_function_unbounded(model, batch, theta, VarZtrx, beta)

def pac_unbounded(model, batch, theta, VarZtrx, rho_params, rng, N= 10):
    """
    estimate of the PAC Bayesian bound (without the empirical error) using several realizations drawn from rho
    """
    m = batch[-1].shape[0]
    realizations = dist_sample(rho_params, N, rng)
    bound = target_function_unbounded_without_empRisk(model,batch,theta,VarZtrx,realizations)
    constants = theta*(1+1/model.delta)+jnp.log(1/model.delta)/jnp.sqrt(m)+VarZtrx/(2*jnp.sqrt(m))
    bound += constants
    bound += target_function_unbounded_KL(model.pi_params,rho_params,m)
    return bound
















