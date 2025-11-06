import numpy as np
from scipy.io import loadmat
from jax import grad, jit
from jax import lax
from jax import random
import jax
from jax import jit
import chex
from functools import partial
import jax.numpy as jnp
import optax
from STOUNewSetup import *
from scipy.stats import  chisquare #kstest as kstest
from scipy.stats import randint
from scipy.stats import kstest
from tqdm import trange,tqdm
from optax._src import wrappers,utils
from optax.monte_carlo import stochastic_gradient_estimators as sge
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'#,2,3'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'



def get_simulated_data(filename):
    data=loadmat(filename+'.mat')
    data=data["data"]

    #data=data-np.mean(data)
    #data=data/np.std(data)

    return data

def rescalingU1(d,eps=0):
  m=np.amin(d) 
  M=np.amax(d)
  #p=(1-2*eps)/(M-m)
  #q=(M*eps-(1-eps)*m)/(M-m)
  p=2/(M-m)
  q=(m+M)/(m-M)
  return d*p+q,p,q

def rescalingU0(d,eps=0):
  m=np.amin(d)
  M=np.amax(d)
  p=(1-2*eps)/(M-m)
  q=(M*eps-(1-eps)*m)/(M-m)
  return d*p+q,p,q

  
def rescalingInv(d,slope,q,eps=0):

  return (d - q)/slope


class MyMultiNormalDiagFromLogScale:	
#to be placed in a specific scritp (collect other distros?)
#rewrite class wrt to functional syntax
  """MultiNormalDiag which directly exposes its input parameters."""

  def __init__(self, loc, nu,seed):
    self._var = jnp.exp(nu)
    self._log_scale = nu /2
    self._mean = loc
    self._param_shape = self._mean.shape
    self.seed=seed
    

  def sample(self, size):
    #print(self._param_shape)
    subkeys=jax.random.split(self.seed,num=size)
    #print(subkeys)
    sample_shape = self._param_shape
    #print(size,sample_shape)
    #print(jax.random.normal(key, shape=sample_shape).shape,self._mean.shape)
    sam=jax.vmap(lambda k : jax.random.normal(k, shape=sample_shape) * jnp.exp(self._log_scale) + (self._mean) )(subkeys)
    #sam=jax.random.multivariate_normal(seed,self._mean,self._scale, size)
    #print(sam.shape)
    return sam

  def log_prob(self, x):
    log_prob = jax.scipy.stats.multivariate_normal.logpdf(x,mean=self._mean, cov=jnp.diag(self._scale))
    # Sum over parameter axes.
    #print('\n\t\tll ',log_prob)
    sum_axis = [-(i + 1) for i in range(len(self._param_shape))]
    #print(sum_axis,len(self._param_shape))
    return jnp.sum(log_prob)#, axis=sum_axis) TURN BACK ON FOR MORE DIMENSIONAL PARAMETERS

  @property
  def log_scale(self) -> chex.Array:
    return self._log_scale

  @property
  def params(self):
    return [self._mean, self._log_scale]


def my_multi_normal(
    key,*params,
) :
  return MyMultiNormalDiagFromLogScale(loc=params[0],nu=params[1],seed=key)#, scale=jnp.diag(params[1]))

def sge_pwj(function,params,dist_builder,rng,num_samples=1):
  #subkeys=jax.random.split(self.seed,num=num_samples)
  #print(subkeys)
  def surrogate(params):
      # We vmap the function application over samples - this ensures that the
      # function we use does not have to be vectorized itself.
      dist = dist_builder(rng,*params)
      eu=jax.vmap(function)(dist.sample((num_samples,)))
      #print(eu)
      return jnp.mean(eu)

  val=surrogate(params)
  grad=jax.grad(surrogate)(params)
  return [val,grad]

def sge_pwj_2(function,params,dist_builder,rng,num_samples=1):
  #subkeys=jax.random.split(self.seed,num=num_samples)
  #print(subkeys)
  def surrogate(params):
      # We vmap the function application over samples - this ensures that the
      # function we use does not have to be vectorized itself.
      dist = dist_builder(rng,*params)
      return (jax.vmap(function)(dist.sample((num_samples,))))
  
  return jax.jacfwd(surrogate)(params)



def l_empirical_risk(A,b,arch,mask,dim,eps):

    #print(eps)
    #print(mask,arch)
    return lambda beta: empirical_risk(A,b,beta,arch,mask,dim,eps)

def r_empirical_risk(A,b,arch,mask,dim,eps):

    return lambda beta: empirical_val(A,b,beta,arch,mask,dim,eps)


def empirical_risk(A,b,realization,arch,mask,dim,eps,bounded=True):
    
    #print(realization.shape) 
    #print(mask)
    jax.debug.print("realization = {}", realization)
    jax.debug.print("realizationshape = {}", realization.shape)
    def pozzo(params,mask,arch):
     #print(params)
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo

    #print(mask,type(mask))
    #print(arch)
    realization=pozzo(realization,mask,arch)
    
    #print(realization[-1].shape)
    #print(A.shape)
    #print('realShape',realization.shape)
    empR=lambda beta : get_loss_function(A,b,beta,eps,bounded)
    #realization=post(rhoParams)#,num_realizations=20)
    #print(realization.shape)
    eU=empR(realization)
    #print(eU.shape)
    #c=np.mean(eU)
    #print("n_c: ",c).
    #print(type(eU))
    #jax.debug.print("A{}", A)
    #jax.debug.print("b: {}", b)
    return eU

def empirical_val(A,b,realization,arch,mask,dim,eps,bounded=True):
    
    #print(realization.shape)          
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo
    masking= lambda a: pozzo(a,mask,arch)
  
    realization=jax.vmap(masking)(realization)
    empR=lambda beta : return_loss_function(A,b,beta,eps,bounded)
    #realization=post(rhoParams)#,num_realizations=20)
    #print(realization.shape)
    eU=empR(realization)
    return eU



#does it make sense to differentiate prior and posterior, when both are gaussian

def get_loss_function(A,b,weights,eps,bounded=True):


    r_eps=0
    #print(arch)
    #print(A.shape)
    #fun_sq = lambda beta: ffnnLossJ(A,arch,beta,b) 

    fun_map = lambda beta: ffnnLossPozzo(A,beta,b,eps,bounded) 

    #fun_b = lambda x: (x <= eps).astype(dtype='float32') * x + (x > eps).astype(dtype='float32') * eps
    
    r_eps= fun_map(weights)#*m#ABS BOUNDED!

    
    return r_eps
    
 
def return_loss_function(A,b,weights,eps,bounded=True):

    r_eps=0

    #print(weights.shape)
    #fun_sq = lambda beta: ffnnLoss(A,arch,beta,b) 

    fun_map = lambda beta: ffnnLossPozzo(A,beta,b,eps,bounded) 

    
    r_eps= jax.vmap(fun_map)(weights)#jax.lax(fun_map,weights,batch_size=b.shape[0]/10)))
    #print(r_eps.shape)
    #jax.lax.map(f, xs, *, batch_size=None)
    return r_eps	#weights.shape keeps track of the sample size of the monte carlo estimator



def dimComp(archs):
	
	dim=0
	#dim=inp*(archs[0]+1)
	for i in range(len(archs)-1):
		dim = dim + (archs[i]*(archs[i+1])+archs[i+1])
	return dim



def post(rhoP,num_realizations=1,seed=1):
    
    '''
    The parameters of the neural network are stored in a linear vector, to prevent memory fill.
    The random seed is already a random key here
    The vector contains the mean and the log scale
    '''
    
    sample_shape = (num_realizations,*rhoP[0].shape)#tuple(num_realizations) + rhoP[0].shape jax.random.key(seed)
    sam=jax.random.normal(seed, shape=sample_shape) * jnp.exp(rhoP[1]/2) + rhoP[0]

    return sam
    
def prior(piP,num_realizations=1,seed=0):

    #print('prior shape',len(piP))
    sample_shape = (num_realizations,*piP[0].shape)#tuple(num_realizations) + rhoP[0].shape
    sam=jax.random.normal(jax.random.key(seed), shape=sample_shape) * jnp.exp(piP[1]/2) + piP[0]

    return sam	

def truePAC(params,piParams,eps,Lip,delta,a_p_c,m,alph,p,dim,arch,chi):

    p=1
    bb=0
    print(eps,Lip,delta,a_p_c,m,alph,p,dim)
    rhoParams=[params[0],params[1]]#jnp.diag(params[1])]
    piParams=[piParams[0],piParams[1]]#jnp.diag(piParams[1])]
    NNsize=dimComp(arch)
    bb=2*(jnp.log(delta))/jnp.sqrt(m) + (.5*eps**2)/jnp.sqrt(m)
    kl=KLdiag_from_log_scale(piParams,rhoParams,NNsize)
    theta=(Lip*a_p_c + 1 )
    #chi= chi2_diag_gaussians(piParams,rhoParams)
    bb+= (1./jnp.sqrt(m))*kl + jnp.sqrt(theta/(2*m)*chi)
    return bb


def pacBound(params,piParams,eps,Lip,delta,a_p_c,a,m,alph,lambda_,p,dim,arch):
    p=1
    bb=0
    rhoParams=[params[0],params[1]]#jnp.diag(params[1])]
    piParams=[piParams[0],piParams[1]]#jnp.diag(piParams[1])]
    NNsize=dimComp(arch)
    #print(NNsize)
    bb= 2*(jnp.log(delta))/jnp.sqrt(m) + (.5*eps**2)/jnp.sqrt(m)
    kl=KLdiag_from_log_scale(piParams,rhoParams,NNsize)
    theta=(Lip*a_p_c+1)  #lambda gamma: 
    bb+= (1./jnp.sqrt(m))*kl+jnp.sqrt(theta/m*kl)
    return bb

def pacBoundE(params,piParams,eps,Lip,delta,a_p_c,a,m,alph,lambda_,p,dim,arch):
    p=1
    bb=0
    rhoParams=[params[0],params[1]]#jnp.diag(params[1])]
    piParams=[piParams[0],piParams[1]]#jnp.diag(piParams[1])]
    NNsize=dimComp(arch)
    bb= 2*(jnp.log(delta))/jnp.sqrt(m) + (.5*eps**2)/jnp.sqrt(m)
    kl=KLdiag_from_log_scale(piParams,rhoParams,NNsize)
    theta=(Lip*a_p_c+1)*alph*jnp.exp(-lambda_*(a-p))  #lambda gamma: 
    bb+= (1./jnp.sqrt(m))*kl+jnp.sqrt(((eps*delta*theta/m)*2*kl))
    return bb

def KLdiag_grad(piParams,rhoParams,NNsize):

    piParams0=piParams[0]#[:NNsize]
    piParams1=piParams[1]#[:NNsize,:NNsize]
    rhoParams0=rhoParams[0]#[:NNsize]
    rhoParams1=rhoParams[1]#[:NNsize,:NNsize]

    grad = lambda pip0,pip1,rhop0,rhop1: jnp.array([(rhop0-pip0)/pip1,.5*(1./pip1-1./rhop1)]) #(rhop0-pip0)/pip1

    gradient= jax.vmap(grad)(piParams0,piParams1,rhoParams0,rhoParams1)
    #print('grad shape',gradient.shape)
    return jnp.transpose(gradient)

def pac_gradient(piParams,rhoParams,NNsize,m,Lip,a_p_c):

    kl_grad= KLdiag_grad(piParams,rhoParams,NNsize)
    return kl_grad*(1/jnp.sqrt(m) + .5*jnp.sqrt((Lip*a_p_c+1)/(m*KLdiag_from_log_scale(piParams,rhoParams,NNsize))))

def pac_approx(piParams,rhoParams,NNsize,m,Lip,a_p_c):

    return KLdiag_from_log_scale(piParams,rhoParams,NNsize)*1./jnp.sqrt(m) + jnp.sqrt((Lip*a_p_c+1)*KLdiag_from_log_scale(piParams,rhoParams,NNsize)/m)

def target_func_unbounded(piParams,rhoParams,NNsize,m):
   return KLdiag(piParams,rhoParams,NNsize)/jnp.sqrt(m) + target_func_unbouded_sampling_from_rho() # ist KLdiag richtig oder from_logscale???

def target_func_unbounded_KL(piParams,rhoParams,NNsize,m):
   return KLdiag_from_log_scale(piParams,rhoParams,NNsize)/jnp.sqrt(m)
   

def target_func_unbouded_sampling_from_rho(A,b,realization,rhoParams,dim,arch,mask,theta, VarZtrx,m): 
  tf = empirical_risk(A,b,realization,arch,mask,0,0,bounded=False)

  def pozzo(params,mask,arch):
      pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
      return pozzo
  realization=pozzo(realization,mask,arch)
  hX = ffnnPozzo(A,realization)
  apc = A.shape[0]
  Liph = LipC(rhoParams,dim,mask,arch) # evtl parallelisieren
  E_rho_Liph = jnp.mean(Liph)
  E_rho_Liph_sq = jnp.mean(jnp.power(Liph,2))
  abs_E_hX = [jnp.abs(jnp.mean(hx)) for hx in hX]
  E_rho_abs_E_hX_Liph = jnp.mean(jnp.multiply(abs_E_hX, Liph))
  E_rho_hXsq = jnp.mean(jnp.power(abs_E_hX,2))
  tf += apc*E_rho_Liph*(theta + VarZtrx)
  tf += E_rho_Liph_sq*apc**2/(2*jnp.sqrt(m))*(VarZtrx+theta**2)
  tf += apc * theta / jnp.sqrt(m) * E_rho_abs_E_hX_Liph
  tf += E_rho_hXsq/(2*jnp.sqrt(m))

  return tf

def tf_unbounded(A,b,rhoParams,dim,arch,mask,theta, VarZtrx,m):
   return lambda beta: target_func_unbouded_sampling_from_rho(A,b,beta,rhoParams,dim,arch,mask,theta, VarZtrx,m)

def pac_unbounded(): # TODO ist target_func_unbounded plus die Konstanten
   pass

def pac_mc_allester(piParams,rhoParams,NNsize,m,Lip,a_p_c,delta):

    return jnp.sqrt( (KLdiag_from_log_scale(piParams,rhoParams,NNsize) + jnp.log(m*delta)) /(2*(m-1))  ) + jnp.sqrt((Lip*a_p_c+1)*KLdiag_from_log_scale(piParams,rhoParams,NNsize)/m)

def pac_reparametrized(piParams,rhoParams,NNsize,m,Lip,a_p_c):

    return KLdiag_from_log_scale(piParams,rhoParams,NNsize)*(1+jnp.sqrt(Lip*a_p_c+1))/jnp.sqrt(m)
  

def chi2_diag_gaussians(piParams,rhoParams):
    """
    mu_p, mu_q: arrays shape (d,)
    var_p, var_q: arrays shape (d,) variances (>=0)
    returns: scalar chi^2(P||Q) (float or np.inf)
    """
    mu_p = rhoParams[0]
    var_p = rhoParams[1]
    mu_q = piParams[0]
    var_q = piParams[1]
    mu_p = jnp.array(mu_p)
    mu_q = jnp.array(mu_q)
    var_p = jnp.array(var_p)
    var_q = jnp.array(var_q)
    #if not (mu_p.shape == mu_q.shape == var_p.shape == var_q.shape):
    #    raise ValueError("Shapes must match")
    # check existence condition
    #if jnp.any(2*var_q <= var_p): SET BACK
    #    return jnp.inf
    delta2 = (mu_p - mu_q)**2
    denom = var_p * (2*var_q - var_p)
    Ii = var_q / jnp.sqrt(denom) * jnp.exp(delta2 / (2*var_q - var_p))
    prod = jnp.prod(Ii)
    return prod 

  
def KLdiag(piParams,rhoParams,NNsize):
    
    '''
    computes the KL divergence for two multivariate gaussians
    

    Parameters
    ----------
    piP : mean and variance of prior distribution.
    rhoP : mean and variance of posterior distribution.

    Returns
    kl: computation of the divergence
    -------
    None.
    '''
    
    #print('nn',NNsize)
    piParams0=piParams[0]#[:NNsize]
    piParams1=piParams[1]#[:NNsize,:NNsize]
    rhoParams0=rhoParams[0]#[:NNsize]
    rhoParams1=rhoParams[1]#[:NNsize,:NNsize]
    
    inv=lambda beta: 1./beta
    #print(jax.vmap(inv)(piParams1))
    kl=jnp.dot(jax.vmap(inv)(piParams1),rhoParams1) 
    #print('\t1',kl*1)
    kl= kl - NNsize
    #print('\t2',kl*1)
    diff=piParams0-rhoParams0
    
    prod=lambda a,b: a*b
    
    #print(jax.vmap(prod)(jax.vmap(inv)(piParams1),diff).shape)
    
    #print('dot ', jnp.dot(diff,jax.vmap(prod)(jax.vmap(inv)(piParams1),diff)).shape)
    
    kl= kl + jnp.dot(diff,jax.vmap(prod)(jax.vmap(inv)(piParams1),diff)) #matmul
    #print('prod kl',kl)
    
    #print('logrho  ',jax.vmap(jnp.log)(rhoParams1))
    #print('logpi',jnp.sum(jax.vmap(jnp.log)(piParams1)))
    #print('logrho',jnp.sum(jax.vmap(jnp.log)(rhoParams1)))
    #print('rhoParams',rhoParams1)
    kl=kl + jnp.sum(jax.vmap(jnp.log)(piParams1))#jnp.log(jnp.prod(piParams1))
    kl=kl - jnp.sum(jax.vmap(jnp.log)(rhoParams1))#jnp.log(jnp.prod(rhoParams1))
    print('kl end',kl)
    return kl
    

def KLdiag_from_log_scale(piParams,rhoParams,NNsize):
    
    '''
    computes the KL divergence for two multivariate gaussians
    

    Parameters
    ----------
    piP : mean and log scale of prior distribution.
    rhoP : mean and log scale of posterior distribution.

    Returns
    kl: computation of the divergence
    -------
    None.
    '''
    #print(piParams[0],rhoParams[0],NNsize)
    #print('nn',NNsize)
    piParams0=piParams[0]#[:NNsize]
    piParams1=piParams[1]#[:NNsize,:NNsize]
    rhoParams0=rhoParams[0]#[:NNsize]
    rhoParams1=rhoParams[1]#[:NNsize,:NNsize]
  
    inv=lambda beta: 1./beta
    #print(jax.vmap(inv)(piParams1))
    kl= jnp.sum(jnp.exp(rhoParams1-piParams1)-1)
    #print('\t1',kl)
    #kl= kl - NNsize
    #print('\t2',kl)
    diff=piParams0-rhoParams0
    
    prod=lambda a,b: a*b
    
    #print(jax.vmap(prod)(jax.vmap(inv)(piParams1),diff).shape)
    
    #print('dot ', jnp.dot(diff,jax.vmap(prod)(jax.vmap(inv)(piParams1),diff)).shape)
    
    kl= kl + jnp.dot(diff,jax.vmap(prod)(jnp.exp(-piParams1),diff)) #matmul
    #print('prod kl',kl)
    
    #print('logrho  ',jax.vmap(jnp.log)(rhoParams1))
    #print('logpi',jnp.sum(jax.vmap(jnp.log)(piParams1)))
    #print('logrho',jnp.sum(jax.vmap(jnp.log)(rhoParams1)))
    #print('rhoParams',rhoParams1)
    kl=kl + jnp.sum(piParams1)#jnp.log(jnp.prod(piParams1))
    #print('prod kl 2',kl)
    kl=kl - jnp.sum(rhoParams1)#jnp.log(jnp.prod(rhoParams1))
    #print('prod kl 3',kl)
    #print('kl end',kl)
    return kl
    



def LipC(piParams,dim,mask,arch,shard_size=1,N=1):#=int(5e2),N=int(1e3)):

    #print(w.shape)
    #print(arch)
    realizations = prior(piParams,num_realizations=N)
    #print(realizations.shape)
    realizations=realizations.reshape((N//shard_size,shard_size,dim))
  
    def pozzo(params,mask,arch):
     
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo

    parPozzo=lambda alpha: pozzo(alpha,mask,arch)
    #naiveLip = jax.vmap(lambda layer: jnp.linalg.norm(layer,ord=2))
    def naiveLip(weights):
      prod=1
      for el in weights:
        prod*=jnp.amax(jnp.abs(el))           #jnp.linalg.norm(el,ord=2)
      return prod
    #lipMap=jax.vmap(lambda beta: naiveLip(beta))
    L=0
    for el in realizations:
      w=jax.vmap(parPozzo)(el)

      c=jax.vmap(naiveLip)(w)
      #print(c.shape)
      L+= jnp.sum(c)
      
    L=L/N
    
    #c = lipMap(w)
    return L
  
def rescalingInv(d,slope,q,eps=0):
  m=np.amin(d)
  M=np.amax(d)
  #p=(1-2*eps)/(M-m)
  #q=(M*eps-(1-eps)*m)/(M-m)
  return (d - q)/m

   

def get_loss_vector(A,b,w,arch,eps):
    
    fun_map= lambda alpha,gamma : ffnnLossV(alpha,arch,w,gamma)#J
    
    fun_abs = lambda beta: jnp.abs(beta)#ffnnLoss(A,arch,beta,b) 

    fun_b = lambda x: (x <= eps).astype(dtype='float32') * x + (x > eps).astype(dtype='float32') * eps
    
    #fun=lambda : fun_b(fun_map(A,b))

    #r_eps=jax.vmap(fun_abs)()
    
    r_eps =jax.vmap(fun_b)(fun_map(A,b))
    #print(r_eps)
    #print(r_eps)
    #print('eps shape',r_eps.shape)
    
    return r_eps#jnp.hstack([jnp.ones(r_eps.shape[0]),r_eps])

def rescalingInv(d,slope,q,eps=0):
  
  return (d - q)/slope



def error(it_params,A_c_e,b_c_e,dim,arch,mask,piParams,eps,Lip,delta,a_p_c,a,m,alpha,lambda_,p,rng,shard_size=int(5e2),N=int(1e3)):


    MC_train_e = r_empirical_risk(A_c_e,b_c_e,arch, mask, dim, eps)
    realizations = post([it_params[0],it_params[1]],num_realizations=N,seed=rng)
    realizations = realizations.reshape((N//shard_size,shard_size,dim))
    error=0
    for el in realizations:
      e = MC_train_e(el)
      #print(e.shape)
      error += jnp.sum(e)
    error=error/N
    #print(error)
    pac=pacBound(it_params,piParams,eps,Lip,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch)
    return [error,pac]



def coordit_pretraining(Z,preT_coord,preT_d_slicing,a,c,p,arch,mask,dim,eps,init_scale=0.0016,pretraining_learning_rate=0.01):

  # assuming the parallelization already took place
  preT_mapped=jax.vmap(preT_d_slicing)(preT_coord)
  preT_mapped=jnp.reshape(preT_mapped,(len(preT_coord),preT_mapped.shape[-1]))
  #print(preT_mapped.shape)
  
  Ac_preT,bc_preT = Z.get_coneJ(preT_mapped,sizeData=preT_mapped.shape[1],preT=True)

  init_log_scale=jnp.log(init_scale)
  
  initParams = [jnp.zeros(dim),jnp.ones(dim)*init_log_scale] 
  
  # seed=0
  #print(mask)
  w0 = prior(initParams)
  w=w0
  #print('ac pre',Ac_preT.shape)
  optimizer = optax.adam(pretraining_learning_rate)	 

  # Initialize parameters of the model + optimizer.
  
  opt_state = optimizer.init(w)

  #preT_size = data_pretraining.shape[1]
  for e in range(12):
    for i in range(Z.Ncones):
      #pretraining dataset size might also be needed 
      #weights must be masked
      scorf = l_empirical_risk(Ac_preT[:,i].reshape((Ac_preT[:,i].shape[0],1)),bc_preT[i],arch, mask, dim, eps)
      
      grads =jax.vmap(jax.grad(scorf))(w)
      updates, opt_state = optimizer.update(grads, opt_state)
      w = optax.apply_updates(w, updates)
  preT_val=jax.vmap(scorf)(w)
  print(preT_val)
  return w,w0,preT_val

def makeh5(net,hdata,names):
	
  #for i in range(len(datasets)):
  #x=net.create_group(str(dataset))
  for j in range(len(names)):
    net.create_dataset(names[j],data=hdata[j])
  #file_.close()



#@jit
def Optimization(file_m,Z,x_size,preT,rescaling,eps,delta,data,inp,p,c,arch,dim,Ndraws,Ncones,Ncones_test,Ncoords,shard_size,lr,rhoScaling,slope=1,q=0,epochs=1,maxit=jnp.inf,piScaling=1,acrit=.25):

    struct=[inp,*arch]
    #print('MC reduced, low epochs,alternative initialization (ones,ones/2)')
    #print('no stopping gradient, pericolo di morte')
    print('number of epochs: ', epochs)
    print('maximum of ',maxit,' iterations')
    #print('NO BIAS')
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
       #print(mask)

    if rescaling:
      print('rescaling between [-1,1]')
      data,pinv,qinv=rescalingU1(data)
      file_m.create_dataset('slope',data=pinv)
      file_m.create_dataset('quota',data=qinv)
    #print('minmax',np.amin(data),np.amax(data))
    data_test=data[:,-(Ncones_test-1)*Z.a:] 
    #print('data test',data_test.shape)
    data_train=data[:,:-Z.a*Ncones_test]
    data_pretraining = data_train[:,-1001*Z.a:] # extra cone for the last point 
    #print(data_pretraining.shape)
    file_m.create_dataset('data_test',data=data_test)
  
    N=data_train.shape[1]
    x_size=data_train.shape[0]
    #print('in otp',x_size)
    list_windows= jnp.array([jnp.arange(element-c*p,element+c*p+1) for element in range(c*p,x_size-c*p)])
    preT_d_slicing= lambda beta: lax.dynamic_index_in_dim(data_pretraining,beta,axis=0) 
    #print('in sgd',x_size)
    #time.sleep(10)
    
	#trange(Ndraws, position=0, desc="r", leave=True, colour='green'):

    pretraining_mean= jnp.zeros(dim)
    # PRETRAINING
    if preT:
      print('pretraining')
      pretraining = jax.vmap(lambda preT_coord: coordit_pretraining(Z,preT_coord,preT_d_slicing,Z.a,Z.c_,Z.p,[inp,*arch],mask,dim,eps))
      
      pretraining_mean,w0,preT_emp_risk = pretraining(list_windows) 
      print('saving preT_ER')
      file_m.create_dataset('pretraining ER',data=preT_emp_risk)
    #print('w0',w0[0,:,:].shape)
    # PAC-BAYES BOUND OPTIMIZATION
  
    coord = []
    rng = jax.random.key(0)
    #output = np.array(np.zeros((Ncoords,Ndraws)))
    output=[]
    for t in reversed(range(p)): #parallelize!!!
        coord.append(jnp.array([[v, - (t + 1)] for v in range(-c*(t + 1), c*(t + 1) + 1)]))
    coord = jnp.concat(coord, axis=0)
    coord = jnp.expand_dims(jnp.expand_dims(coord, 0), 0)
    
    bounds=[]
    pars=[]
 
    aux_in=[inp,*arch[:-1]]

    piScale=jnp.array([])
    for in_,out_ in zip(aux_in,arch):
      layer=jnp.ones((in_+1)*out_)/in_
      piScale=jnp.hstack([piScale,layer])


    if piScaling!=1:
      print('fixed setup, posterior with zero mean')
      piScale=jnp.ones(dim)*piScaling
    else: 
        print('mixed setup, post erior with zero mean')
    
    if preT:
      
      pretraining_fun = jax.vmap(lambda beta: jnp.vstack([beta,jnp.ones(dim)*jnp.linalg.norm(beta,ord=1)]))
      pretraining_parameters = pretraining_fun(pretraining_mean)
      #pretraining_parameters = list(pretraining_parameters.reshape(((x_size-2*p*c)//shard_size,shard_size,2,dim))))
      params=[pretraining_parameters[el*shard_size:(el+1)*shard_size] for el in range((x_size-2*p*c)//shard_size) ]
      #print('preT_params',params)
      #pretraining_fun_prior = jax.vmap(lambda beta: jnp.vstack([beta,jnp.ones(dim)*piScale]))
      #piParams_preT = pretraining_fun_prior(w0)#SET BACK GRID SEARCH INIT!!!
      #piParams = [piParams_preT[el*shard_size:(el+1)*shard_size] for el in range((x_size-2*p*c)//shard_size) ]
      piParams = [w0[0,0,:],piScale]
      
    else:
      rhoParams=[jnp.zeros(dim),jnp.ones(dim)*jnp.log(.25)] #jnp.ones(dim)/2
      sharded_params=jax.vmap(lambda dummy: jnp.vstack(rhoParams))(range(shard_size))
      params= [sharded_params for el in range((x_size-2*p*c)//shard_size)]   
      piParams=[jnp.zeros(dim),piScale]
      
    start_learning_rate = jnp.float32(lr)

    print('computing Lipschitz constant... ')
    Lip= LipC(piParams,dim,mask,[inp,*arch])
    print(Lip)
    #jax.clear_caches()

    Acones_dummy = jnp.zeros((Ncones,inp))
    bcones_dummy = jnp.zeros(Ncones)
    
    sharded_Acones = jax.vmap(lambda dummy: jnp.vstack(Acones_dummy))(range(shard_size))
    Acones_s = [sharded_Acones for el in range((x_size-2*p*c)//shard_size)] 

    sharded_bcones = jax.vmap(lambda dummy: jnp.vstack(bcones_dummy))(range(shard_size))
    bcones_s = [sharded_bcones for el in range((x_size-2*p*c)//shard_size)] 

    #print('shape in sharding ')
    #print(sharded_Acones.shape)
    #print(sharded_bcones.shape)
    
    #OPTIMIZER
    #opt_states=[]
    optimizer = optax.adam(start_learning_rate)	#SGD ON
    #opt_state=optimizer.init(jnp.vstack(rhoParams))
    #for el in range(x_size):
    #  opt_states.append(opt_state)
    # Vectorized optimizer state init
    if preT:
      opt_state_grid= [jax.vmap(optimizer.init)(el) for el in params]
    else:
      opt_state_grid = [jax.vmap(optimizer.init)(sharded_params) for el in range((x_size-2*p*c)//shard_size)]
    #print(opt_state_grid[0])
    #for x_coord in range(Ncoords):#,40):#(x_size, position=0,leave=True, desc="coordinate", colour='red'):    
    #@partial(jit,static_argnums=1)
    time_windows=([jnp.arange(jnp.maximum(0,Z.N-Z.a*Z.Ncones*(element+1)),(Z.N-Z.a*Z.Ncones*element)) for element in range(Z.Nbatches)])
    t_slicing= lambda beta: lax.dynamic_index_in_dim(jnp.transpose(data_train), beta,axis=0)
    #params_shards= jnp.array([jnp.arange(element,(element+shard_size)) for element in range(p,x_size-p-1,shard_size)])
    list_shards= ([jnp.array([jnp.arange(c_coord-p*c,c_coord+p*c+1) for c_coord in range(element,element+shard_size)]) for element in range(p*c,x_size-p*c,shard_size)])
    #print('N batches=',Z.Nbatches)

    def multi_ef(it_coord,params,a,inp,arch,mask,Ndraws,Ncones_test,rng):

        test_mapped=jax.lax.map(test_slicing,(it_coord))#print('coord',x_coord)
        #print('shape',test_mapped.shape,batch)
        test_mapped=jnp.reshape(test_mapped,(len(it_coord),a*Ncones_test))
        #print(test_mapped.shape)
        Ac,bc=Z.get_coneJ((test_mapped),sizeData=test_mapped.shape[1])
        #print(Ac.shape)
        #w=post([params[0],params[1]],Ndraws,seed=rng)    
        #print(w.shape)
        #out=jax.vmap(lambda A : ffnnV(A,[inp,*arch],mask,w))
        #m_e_f = out(jnp.transpose(Ac))
        #m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        #print('mef',m_e_f.shape)
        
        
        return [Ac,bc] #m_e_f,

    def coordit(it_coord,it_params,opt_state,batch,Bsize,m,dim,Lip,inp,rng=rng):
        #if not x_coord%10: print('cood: ',x_coord)
        #x_coord=jnp.where(it_coord<Ncoords-p,it_coord,0)
        #Z.data=data
        #print(it_params.shape[0])
        #it_params=jnp.reshape(it_params,(2,int(it_params.shape[0]/2)))
        #print('in init coord')   ####################################################
        #print(it_coord)
        window_mapped=jax.vmap(d_slicing)(it_coord)#print('coord',x_coord)
        #print('shape',window_mapped.shape)
        window_mapped=jnp.reshape(window_mapped,(len(it_coord),Bsize))
        #print('shape',window_mapped.shape)
        
        #print(cone_mapped.shape)
        #print('shape',cone_mapped.shape)
        Acones,bcones=Z.get_coneJ((window_mapped),sizeData=time_windows[batch].shape[0])
        #print('shape in coordit ',Acones.shape,bcones.shape)
        input_size,Z.m=Acones.shape
        
        sgds=[]#np.zeros((Nsteps,2,len(rhoParams[0])))
        sges=[]#np.zeros(Nsteps)					
        #print(eps)
        scorf=l_empirical_risk(Acones[:,:],bcones[:],([inp,*arch]),mask,dim,eps) #jax.jit
        scorfval=r_empirical_risk(Acones[:,:],bcones[:],([inp,*arch]),mask,dim,eps)
        #grads= pac_gradient(piParams,it_params,dim,m,Lip,inp)

        pac_mapped = lambda beta: pac_approx(piParams,beta,dim,m,Lip,inp)
        
        #pac_mapped= lambda beta : pac_mc_allester(piParams,beta,dim,m,Lip,inp,delta)
        
        val_grad=pac_mapped(it_params)
        grad= jax.grad(pac_mapped)(it_params)
        #print(KLdiag_from_log_scale(piParams,it_params,dim))
        #value,grads = jax.value_and_grad(funApprox)(it_params) #EXPERIMENTAL VERSION!!! CHANGE
        val_jest,jest = sge_pwj(scorf,it_params,my_multi_normal,rng)
        #print('\n',jest,grad) 
        #jest = jnp.mean((jest),axis=0)
        #print('1',jest)
        #jest2 = sge_pwj_2(scorf,it_params,my_multi_normal,rng,num_samples=5)
        #jest2= jnp.mean(jest2,axis=0)
        #print('2',jest2)
        #rint(jnp.allclose(jest,jest2))
        #print('diff',jest-jest2)
        updates = jest+grad
        #print(updates)#print(updates)
        #print(opt_state)
        updates, opt_state = optimizer.update(updates,opt_state,it_params) #print(opt_state)
        #print('jest',jest)	
        it_params = optax.apply_updates(it_params,updates)
        #print(it_params,jnp.prod(it_params[0]),jnp.prod(it_params[1]))
        
        return [it_params,opt_state,Acones,bcones,val_jest,val_grad]

    
    countit=0
    #best_params=[]
    min_it=0
    min_error=jnp.inf
    milestone_seeds=[]
    val_jest_epoch= jnp.empty((Z.Nbatches,x_size-2*Z.c_*Z.p))
    val_grad_epoch= jnp.empty((Z.Nbatches,x_size-2*Z.c_*Z.p))
    #print(Z.Nbatches)
    for epoch in trange(1,epochs+1, desc='epochs', colour='green'): #range(1,epochs+1):
        #print('EPOCH: ', epoch)
        file_epoch= file_m.create_group('epoch'+str(epoch))
        val_jest_batch= jnp.empty((0,x_size-2*Z.c_*Z.p))
        val_grad_batch= jnp.empty((0,x_size-2*Z.c_*Z.p))
      
        for batch in range(Z.Nbatches): #trange(1, Z.Nbatches, desc='batching', colour='red'):
            #key = jax.random.PRNGKey(batch)
            countit+=1
            #print('in batch')#,countit)
            if countit>maxit: break
            key = jax.random.PRNGKey(batch+Z.Nbatches*(epoch-1))
            keys = [jax.random.split(key*(el+1), shard_size).reshape(shard_size, 2) for el in range((x_size-2*p*c)//shard_size)]
            time_mapped=jax.vmap(t_slicing)(time_windows[batch])
            time_mapped=jnp.squeeze(time_mapped,axis=1)     
            shard_slicing = lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)
            d_slicing = lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)  #  [,:]
            
            ccc = jax.vmap(lambda og,pmap,opt,rngM: coordit(og,pmap,opt,batch,Z.Ncones*Z.a,Z.Ncones,dim,Lip,inp,rngM),in_axes=(0,0,0,0))
    
            output = jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(Ndraws)))
            #Ac_stacked = jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(inp)))
            #bc_stacked = pit=jnp.array([])
              
            
            val_jest_stacked= jnp.array([])
            val_grad_stacked= jnp.array([])
            params_stacked =jnp.transpose(jax.vmap(lambda dummy:jnp.array([[],[]]))(range(dim)))
            for ls_i in range((x_size-2*p*c)//shard_size):
              #print('a')
              [params[ls_i],opt_state_grid[ls_i],Acones_s[ls_i],bcones_s[ls_i],val_jest,val_grad]= ccc(list_shards[ls_i], params[ls_i], opt_state_grid[ls_i], keys[ls_i])#jax.jit
              
              params_stacked=jnp.vstack([params_stacked,params[ls_i]])
              val_jest_stacked=jnp.hstack([val_jest_stacked,val_jest])
              val_grad_stacked=jnp.hstack([val_grad_stacked,val_grad])

            val_jest_batch= jnp.vstack([val_jest_batch,val_jest_stacked])
            val_grad_batch= jnp.vstack([val_grad_batch,val_grad_stacked])
      
        val_jest_epoch=jnp.vstack([val_jest_epoch,val_jest_batch])
        val_grad_epoch=jnp.vstack([val_grad_epoch,val_grad_batch])

        if countit>maxit: break
        #print('out of b')
        # assuming each epoch is a milestone epoch 
        milestone_train_error = jnp.array([])
        #multi_e_f_stacked=jnp.empty((0,Ndraws,Ncones_test-1))
        b_c_e_stacked=jnp.empty((0,Ncones_test-1))
        test_error = jnp.array([])
        bound_train = jnp.array([])
        bound_test = jnp.array([])
      
        milestone_seeds.append(jax.random.key(epoch))

        test_slicing = lambda beta: lax.dynamic_index_in_dim(data_test,beta,axis=0)
        finalStep=lambda coord,beta: multi_ef(coord, beta,Z.a,inp, arch,mask,Ndraws,Ncones_test-1,milestone_seeds[-1])

#(it_params,A_c_e,b_c_e,dim,arch,mask,piParams,eps,Lip,delta,a_p_c,a,m,alpha,lambda_,p,rng,num_realization=int(1e3)):
        #print((x_size-2*p*c)//shard_size)
        e_mapped_train= jax.vmap(lambda  it_pars,it_A,it_b: error(it_pars,it_A,it_b,dim,[inp,*arch],mask,piParams,eps,Lip,delta,inp,Z.a,Z.Ncones,Z.alpha,Z.lambda_,Z.p,milestone_seeds[-1]))
        e_mapped_test = jax.vmap(lambda  it_pars,it_A,it_b: error(it_pars,it_A,it_b,dim,[inp,*arch],mask,piParams,eps,Lip,delta,inp,Z.a,Ncones_test-1,Z.alpha,Z.lambda_,Z.p,milestone_seeds[-1]))
        for ls_i in range((x_size-2*p*c)//shard_size): #might opt for an alternative shard_size
              mile_train_e,pac_train = e_mapped_train(params[ls_i],Acones_s[ls_i],bcones_s[ls_i])
              milestone_train_error=jnp.hstack([milestone_train_error,mile_train_e])
              bound_train = jnp.hstack([bound_train,pac_train])
          
              A_c_e,b_c_e=jax.vmap(finalStep)(list_shards[ls_i],params[ls_i]) #save b_c_e_stacked
              #multi_e_f_stacked=jnp.vstack([multi_e_f_stacked,multi_e_f])
              b_c_e_stacked=jnp.vstack([b_c_e_stacked,b_c_e])
              test_e,pac_test= e_mapped_test(params[ls_i],A_c_e,b_c_e)
              test_error=jnp.hstack([test_error,test_e])   
              bound_test = jnp.hstack([bound_test,pac_test])  

        #print(bound_train.shape)
        if jnp.mean(bound_train+milestone_train_error,axis=0)<min_error: 
          #print('new minimum',min_error)
          min_it=epoch
          min_error=jnp.mean(bound_train+milestone_train_error,axis=0)
          best_params=params_stacked
        #print(bound_test)      
        hdata=[Z.a,Z.lambda_, Z.Ncones,bound_train,milestone_train_error,test_error,bound_test,val_jest_batch,val_grad_batch,b_c_e_stacked]
        names=[ 'a','lambda', 'm', 'bound_train', 'train_errors','test_errors','bound_test','val_jest','val_grad','cones_test']
        makeh5(file_epoch,hdata,names)

    file_last=file_m.create_dataset('last params',data=params_stacked)
    
    file_best=file_m.create_group('min')
    file_best.create_dataset('min_error',data=min_error)
    file_best.create_dataset('min iteration',data=min_it)
    file_best.create_dataset('best params',data=best_params)
        

def Optimization_unbounded(file_m,Z,x_size,preT,rescaling,eps,delta,data,inp,p,c,arch,dim,Ndraws,Ncones,Ncones_test,Ncoords,shard_size,lr,rhoScaling,slope=1,q=0,epochs=1,maxit=jnp.inf,piScaling=1,acrit=.25):

    struct=[inp,*arch]
    #print('MC reduced, low epochs,alternative initialization (ones,ones/2)')
    #print('no stopping gradient, pericolo di morte')
    print('number of epochs: ', epochs)
    print('maximum of ',maxit,' iterations')
    #print('NO BIAS')
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
       #print(mask)

    if rescaling:
      print('rescaling between [-1,1]')
      data,pinv,qinv=rescalingU1(data)
      file_m.create_dataset('slope',data=pinv)
      file_m.create_dataset('quota',data=qinv)
    #print('minmax',np.amin(data),np.amax(data))
    data_test=data[:,-(Ncones_test-1)*Z.a:] 
    #print('data test',data_test.shape)
    data_train=data[:,:-Z.a*Ncones_test]
    data_pretraining = data_train[:,-1001*Z.a:] # extra cone for the last point 
    #print(data_pretraining.shape)
    file_m.create_dataset('data_test',data=data_test)
  
    N=data_train.shape[1]
    x_size=data_train.shape[0]
    #print('in otp',x_size)
    list_windows= jnp.array([jnp.arange(element-c*p,element+c*p+1) for element in range(c*p,x_size-c*p)])
    preT_d_slicing= lambda beta: lax.dynamic_index_in_dim(data_pretraining,beta,axis=0) 
    #print('in sgd',x_size)
    #time.sleep(10)
    
	#trange(Ndraws, position=0, desc="r", leave=True, colour='green'):

    pretraining_mean= jnp.zeros(dim)
    # PRETRAINING
    if preT:
      print('pretraining')
      pretraining = jax.vmap(lambda preT_coord: coordit_pretraining(Z,preT_coord,preT_d_slicing,Z.a,Z.c_,Z.p,[inp,*arch],mask,dim,eps))
      
      pretraining_mean,w0,preT_emp_risk = pretraining(list_windows) 
      print('saving preT_ER')
      file_m.create_dataset('pretraining ER',data=preT_emp_risk)
    #print('w0',w0[0,:,:].shape)
    # PAC-BAYES BOUND OPTIMIZATION
  
    coord = []
    rng = jax.random.key(0)
    #output = np.array(np.zeros((Ncoords,Ndraws)))
    output=[]
    for t in reversed(range(p)): #parallelize!!!
        coord.append(jnp.array([[v, - (t + 1)] for v in range(-c*(t + 1), c*(t + 1) + 1)]))
    coord = jnp.concat(coord, axis=0)
    coord = jnp.expand_dims(jnp.expand_dims(coord, 0), 0)
    
    bounds=[]
    pars=[]
 
    aux_in=[inp,*arch[:-1]]

    piScale=jnp.array([])
    for in_,out_ in zip(aux_in,arch):
      layer=jnp.ones((in_+1)*out_)/in_
      piScale=jnp.hstack([piScale,layer])


    if piScaling!=1:
      print('fixed setup, posterior with zero mean')
      piScale=jnp.ones(dim)*piScaling
    else: 
        print('mixed setup, post erior with zero mean')
    
    if preT:
      
      pretraining_fun = jax.vmap(lambda beta: jnp.vstack([beta,jnp.ones(dim)*jnp.linalg.norm(beta,ord=1)]))
      pretraining_parameters = pretraining_fun(pretraining_mean)
      #pretraining_parameters = list(pretraining_parameters.reshape(((x_size-2*p*c)//shard_size,shard_size,2,dim))))
      params=[pretraining_parameters[el*shard_size:(el+1)*shard_size] for el in range((x_size-2*p*c)//shard_size) ]
      #print('preT_params',params)
      #pretraining_fun_prior = jax.vmap(lambda beta: jnp.vstack([beta,jnp.ones(dim)*piScale]))
      #piParams_preT = pretraining_fun_prior(w0)#SET BACK GRID SEARCH INIT!!!
      #piParams = [piParams_preT[el*shard_size:(el+1)*shard_size] for el in range((x_size-2*p*c)//shard_size) ]
      piParams = [w0[0,0,:],piScale]
      
    else:
      rhoParams=[jnp.zeros(dim),jnp.ones(dim)*jnp.log(.25)] #jnp.ones(dim)/2
      sharded_params=jax.vmap(lambda dummy: jnp.vstack(rhoParams))(range(shard_size))
      params= [sharded_params for el in range((x_size-2*p*c)//shard_size)]   
      piParams=[jnp.zeros(dim),piScale]
      
    start_learning_rate = jnp.float32(lr)

    print('computing Lipschitz constant... ')
    Lip= LipC(piParams,dim,mask,[inp,*arch])
    print(Lip)
    #jax.clear_caches()

    Acones_dummy = jnp.zeros((Ncones,inp))
    bcones_dummy = jnp.zeros(Ncones)
    
    sharded_Acones = jax.vmap(lambda dummy: jnp.vstack(Acones_dummy))(range(shard_size))
    Acones_s = [sharded_Acones for el in range((x_size-2*p*c)//shard_size)] 

    sharded_bcones = jax.vmap(lambda dummy: jnp.vstack(bcones_dummy))(range(shard_size))
    bcones_s = [sharded_bcones for el in range((x_size-2*p*c)//shard_size)] 

    #print('shape in sharding ')
    #print(sharded_Acones.shape)
    #print(sharded_bcones.shape)
    
    #OPTIMIZER
    #opt_states=[]
    optimizer = optax.adam(start_learning_rate)	#SGD ON
    #opt_state=optimizer.init(jnp.vstack(rhoParams))
    #for el in range(x_size):
    #  opt_states.append(opt_state)
    # Vectorized optimizer state init
    if preT:
      opt_state_grid= [jax.vmap(optimizer.init)(el) for el in params]
    else:
      opt_state_grid = [jax.vmap(optimizer.init)(sharded_params) for el in range((x_size-2*p*c)//shard_size)]
    #print(opt_state_grid[0])
    #for x_coord in range(Ncoords):#,40):#(x_size, position=0,leave=True, desc="coordinate", colour='red'):    
    #@partial(jit,static_argnums=1)
    time_windows=([jnp.arange(jnp.maximum(0,Z.N-Z.a*Z.Ncones*(element+1)),(Z.N-Z.a*Z.Ncones*element)) for element in range(Z.Nbatches)])
    t_slicing= lambda beta: lax.dynamic_index_in_dim(jnp.transpose(data_train), beta,axis=0)
    #params_shards= jnp.array([jnp.arange(element,(element+shard_size)) for element in range(p,x_size-p-1,shard_size)])
    list_shards= ([jnp.array([jnp.arange(c_coord-p*c,c_coord+p*c+1) for c_coord in range(element,element+shard_size)]) for element in range(p*c,x_size-p*c,shard_size)])
    #print('N batches=',Z.Nbatches)

    def multi_ef(it_coord,params,a,inp,arch,mask,Ndraws,Ncones_test,rng):

        test_mapped=jax.lax.map(test_slicing,(it_coord))#print('coord',x_coord)
        #print('shape',test_mapped.shape,batch)
        test_mapped=jnp.reshape(test_mapped,(len(it_coord),a*Ncones_test))
        #print(test_mapped.shape)
        Ac,bc=Z.get_coneJ((test_mapped),sizeData=test_mapped.shape[1])
        #print(Ac.shape)
        #w=post([params[0],params[1]],Ndraws,seed=rng)    
        #print(w.shape)
        #out=jax.vmap(lambda A : ffnnV(A,[inp,*arch],mask,w))
        #m_e_f = out(jnp.transpose(Ac))
        #m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        #print('mef',m_e_f.shape)
        
        
        return [Ac,bc] #m_e_f,

    def coordit(it_coord,it_params,opt_state,batch,Bsize,m,dim,Lip,inp,rng=rng):
        #if not x_coord%10: print('cood: ',x_coord)
        #x_coord=jnp.where(it_coord<Ncoords-p,it_coord,0)
        #Z.data=data
        #print(it_params.shape[0])
        #it_params=jnp.reshape(it_params,(2,int(it_params.shape[0]/2)))
        #print('in init coord')   ####################################################
        #print(it_coord)
        window_mapped=jax.vmap(d_slicing)(it_coord)#print('coord',x_coord)
        #print('shape',window_mapped.shape)
        window_mapped=jnp.reshape(window_mapped,(len(it_coord),Bsize))
        #print('shape',window_mapped.shape)
        
        #print(cone_mapped.shape)
        #print('shape',cone_mapped.shape)
        Acones,bcones=Z.get_coneJ((window_mapped),sizeData=time_windows[batch].shape[0])
        #print('shape in coordit ',Acones.shape,bcones.shape)
        input_size,Z.m=Acones.shape
        
        sgds=[]#np.zeros((Nsteps,2,len(rhoParams[0])))
        sges=[]#np.zeros(Nsteps)					
        #print(eps)
        theta = Z.thetatilder
        VarZtrx = Z.truncated_covs_between_all_members_of_cone()[1][0][0]
        scorf=tf_unbounded(Acones,bcones,it_params,dim,arch,mask,theta,VarZtrx,m) #jax.jit
        #scorfval=r_empirical_risk(Acones[:,:],bcones[:],([inp,*arch]),mask,dim,eps)
        #grads= pac_gradient(piParams,it_params,dim,m,Lip,inp)

        kl_mapped = lambda beta: target_func_unbounded_KL(piParams,beta,dim,m) # TODO NNsize scheint nicht benötigt, idk ob dim == NNsize
        
        #pac_mapped= lambda beta : pac_mc_allester(piParams,beta,dim,m,Lip,inp,delta)
        
        val_grad=kl_mapped(it_params)
        grad= jax.grad(kl_mapped)(it_params)
        #print(KLdiag_from_log_scale(piParams,it_params,dim))
        #value,grads = jax.value_and_grad(funApprox)(it_params) #EXPERIMENTAL VERSION!!! CHANGE
        val_jest,jest = sge_pwj(scorf,it_params,my_multi_normal,rng)
        #print('\n',jest,grad) 
        #jest = jnp.mean((jest),axis=0)
        #print('1',jest)
        #jest2 = sge_pwj_2(scorf,it_params,my_multi_normal,rng,num_samples=5)
        #jest2= jnp.mean(jest2,axis=0)
        #print('2',jest2)
        #rint(jnp.allclose(jest,jest2))
        #print('diff',jest-jest2)
        updates = jest+grad
        #print(updates)#print(updates)
        #print(opt_state)
        updates, opt_state = optimizer.update(updates,opt_state,it_params) #print(opt_state)
        #print('jest',jest)	
        it_params = optax.apply_updates(it_params,updates)
        #print(it_params,jnp.prod(it_params[0]),jnp.prod(it_params[1]))
        
        return [it_params,opt_state,Acones,bcones,val_jest,val_grad]

    
    countit=0
    #best_params=[]
    min_it=0
    min_error=jnp.inf
    milestone_seeds=[]
    val_jest_epoch= jnp.empty((Z.Nbatches,x_size-2*Z.c_*Z.p))
    val_grad_epoch= jnp.empty((Z.Nbatches,x_size-2*Z.c_*Z.p))
    #print(Z.Nbatches)
    for epoch in trange(1,epochs+1, desc='epochs', colour='green'): #range(1,epochs+1):
        #print('EPOCH: ', epoch)
        file_epoch= file_m.create_group('epoch'+str(epoch))
        val_jest_batch= jnp.empty((0,x_size-2*Z.c_*Z.p))
        val_grad_batch= jnp.empty((0,x_size-2*Z.c_*Z.p))
      
        for batch in range(Z.Nbatches): #trange(1, Z.Nbatches, desc='batching', colour='red'):
            #key = jax.random.PRNGKey(batch)
            countit+=1
            #print('in batch')#,countit)
            if countit>maxit: break
            key = jax.random.PRNGKey(batch+Z.Nbatches*(epoch-1))
            keys = [jax.random.split(key*(el+1), shard_size).reshape(shard_size, 2) for el in range((x_size-2*p*c)//shard_size)]
            time_mapped=jax.vmap(t_slicing)(time_windows[batch])
            time_mapped=jnp.squeeze(time_mapped,axis=1)     
            shard_slicing = lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)
            d_slicing = lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)  #  [,:]
            
            ccc = jax.vmap(lambda og,pmap,opt,rngM: coordit(og,pmap,opt,batch,Z.Ncones*Z.a,Z.Ncones,dim,Lip,inp,rngM),in_axes=(0,0,0,0))
    
            output = jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(Ndraws)))
            #Ac_stacked = jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(inp)))
            #bc_stacked = pit=jnp.array([])
              
            
            val_jest_stacked= jnp.array([])
            val_grad_stacked= jnp.array([])
            params_stacked =jnp.transpose(jax.vmap(lambda dummy:jnp.array([[],[]]))(range(dim)))
            for ls_i in range((x_size-2*p*c)//shard_size):
              #print('a')
              [params[ls_i],opt_state_grid[ls_i],Acones_s[ls_i],bcones_s[ls_i],val_jest,val_grad]= ccc(list_shards[ls_i], params[ls_i], opt_state_grid[ls_i], keys[ls_i])#jax.jit
              
              params_stacked=jnp.vstack([params_stacked,params[ls_i]])
              val_jest_stacked=jnp.hstack([val_jest_stacked,val_jest])
              val_grad_stacked=jnp.hstack([val_grad_stacked,val_grad])

            val_jest_batch= jnp.vstack([val_jest_batch,val_jest_stacked])
            val_grad_batch= jnp.vstack([val_grad_batch,val_grad_stacked])
      
        val_jest_epoch=jnp.vstack([val_jest_epoch,val_jest_batch])
        val_grad_epoch=jnp.vstack([val_grad_epoch,val_grad_batch])

        if countit>maxit: break
        #print('out of b')
        # assuming each epoch is a milestone epoch 
        milestone_train_error = jnp.array([])
        #multi_e_f_stacked=jnp.empty((0,Ndraws,Ncones_test-1))
        b_c_e_stacked=jnp.empty((0,Ncones_test-1))
        test_error = jnp.array([])
        bound_train = jnp.array([])
        bound_test = jnp.array([])
      
        milestone_seeds.append(jax.random.key(epoch))

        test_slicing = lambda beta: lax.dynamic_index_in_dim(data_test,beta,axis=0)
        finalStep=lambda coord,beta: multi_ef(coord, beta,Z.a,inp, arch,mask,Ndraws,Ncones_test-1,milestone_seeds[-1])

#(it_params,A_c_e,b_c_e,dim,arch,mask,piParams,eps,Lip,delta,a_p_c,a,m,alpha,lambda_,p,rng,num_realization=int(1e3)):
        #print((x_size-2*p*c)//shard_size)
        e_mapped_train= jax.vmap(lambda  it_pars,it_A,it_b: error(it_pars,it_A,it_b,dim,[inp,*arch],mask,piParams,eps,Lip,delta,inp,Z.a,Z.Ncones,Z.alpha,Z.lambda_,Z.p,milestone_seeds[-1]))
        e_mapped_test = jax.vmap(lambda  it_pars,it_A,it_b: error(it_pars,it_A,it_b,dim,[inp,*arch],mask,piParams,eps,Lip,delta,inp,Z.a,Ncones_test-1,Z.alpha,Z.lambda_,Z.p,milestone_seeds[-1]))
        for ls_i in range((x_size-2*p*c)//shard_size): #might opt for an alternative shard_size
              mile_train_e,pac_train = e_mapped_train(params[ls_i],Acones_s[ls_i],bcones_s[ls_i])
              milestone_train_error=jnp.hstack([milestone_train_error,mile_train_e])
              bound_train = jnp.hstack([bound_train,pac_train])
          
              A_c_e,b_c_e=jax.vmap(finalStep)(list_shards[ls_i],params[ls_i]) #save b_c_e_stacked
              #multi_e_f_stacked=jnp.vstack([multi_e_f_stacked,multi_e_f])
              b_c_e_stacked=jnp.vstack([b_c_e_stacked,b_c_e])
              test_e,pac_test= e_mapped_test(params[ls_i],A_c_e,b_c_e)
              test_error=jnp.hstack([test_error,test_e])   
              bound_test = jnp.hstack([bound_test,pac_test])  

        #print(bound_train.shape)
        if jnp.mean(bound_train+milestone_train_error,axis=0)<min_error: 
          #print('new minimum',min_error)
          min_it=epoch
          min_error=jnp.mean(bound_train+milestone_train_error,axis=0)
          best_params=params_stacked
        #print(bound_test)      
        hdata=[Z.a,Z.lambda_, Z.Ncones,bound_train,milestone_train_error,test_error,bound_test,val_jest_batch,val_grad_batch,b_c_e_stacked]
        names=[ 'a','lambda', 'm', 'bound_train', 'train_errors','test_errors','bound_test','val_jest','val_grad','cones_test']
        makeh5(file_epoch,hdata,names)

    file_last=file_m.create_dataset('last params',data=params_stacked)
    
    file_best=file_m.create_group('min')
    file_best.create_dataset('min_error',data=min_error)
    file_best.create_dataset('min iteration',data=min_it)
    file_best.create_dataset('best params',data=best_params)
        
