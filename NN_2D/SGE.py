import numpy as np
from scipy.io import loadmat
from jax import grad, jit
from jax import lax
from jax import random
import jax
from jax import jit
import jax.numpy as jnp
import optax
from STOU import *
from scipy.stats import randint
from tqdm import trange,tqdm
#from optax._src import wrappers,utils
#from optax.monte_carlo import stochastic_gradient_estimators as sge
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
    sam=jax.vmap(lambda k : jax.random.normal(k, shape=sample_shape) * jnp.exp(self._log_scale) + (self._mean) )(subkeys)
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
      eu=jax.vmap(score_function)(dist.sample((num_samples,)))
      return jnp.mean(eu)

  val=surrogate(params)
  grad=jax.grad(surrogate)(params)
  return [val,grad]

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
      return (jax.vmap(function)(dist.sample((num_samples,))))


  
  return jax.grad(surrogate)(params)



def l_empirical_risk(A,b,arch,mask,dim,eps):
    #lambda function for the computation of the empirical risk (to compute the gradient over)
    return lambda beta: empirical_risk(A,b,beta,arch,mask,dim,eps)

def r_empirical_risk(A,b,arch,mask,dim,eps):
    #lambda function for the computation of the empirical risk ()
    return lambda beta: empirical_val(A,b,beta,arch,mask,dim,eps)


def empirical_risk(A,b,realization,arch,mask,dim,eps):

    '''
    Function that computes the empirical risk with respect to the absolute loss
    Input:
        A: cone inputs X_i
        b: cone outputs Y_i
        realization: samples from the newtork distribution
        arch: network layers 
        mask: network architecture structure
        dim: size of the network
        eps: truncation level of the loss function
    Output:
        eU: vector of empirical risk for each single draw from the distribution
    '''
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo
    #reshapes the network parameter according to the layer structure
    realization=pozzo(realization,mask,arch)
    #computes the loss function over all the distribution draws
    empR=lambda beta : ffnn_loss_forward_pass(A,beta,b,eps)
    eU=empR(realization)
    return eU

def empirical_val(A,b,realization,arch,mask,dim,eps):
    
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo
    masking= lambda a: pozzo(a,mask,arch)
    realization=jax.vmap(masking)(realization)
    empR=lambda beta : ffnn_loss_forward_pass(A,beta,b,eps)
    eU=jax.vmap(empR)(realization)
    return eU

def get_loss_function(A,b,weights,eps):
    '''
    Computes the loss function for a single draw from the distribution
    '''
    fun_map = lambda beta: ffnn_loss_forward_pass(A,beta,b,eps) 
    r_eps= fun_map(weights) 
    return r_eps
    
 
def return_loss_function(A,b,weights,eps):
    '''
    Computes the loss funcion for multiple draws from the distribution
    '''
    fun_map = lambda beta: ffnn_loss_forward_pass(A,beta,b,eps) 
    #maps the loss computation over all the distribution draws
    r_eps= jax.vmap(fun_map)(weights) 
    return r_eps	 

def dimComp(archs):
	'''
	Computes the size of the network
	Input:
		archs: layer structure of the network
	'''
	dim=0
	for i in range(len(archs)-1):
		dim = dim + (archs[i]*(archs[i+1])+archs[i+1])
	return dim



def dist_sample(rhoP,num_realizations=1,seed=1):
    
    '''
    Function that samples from a multivariate normal distribution
    Input:
        rhoP: parameters of the distribution to sample from
        num_realizations: number of draws to sample from the distribution
        seed: random PRNG key
    Output:
        Sam: array of the distribution samples
    '''
    
    sample_shape = (num_realizations,*rhoP[0].shape)#tuple(num_realizations) + rhoP[0].shape jax.random.key(seed)
    sam=jax.random.normal(seed, shape=sample_shape) * jnp.exp(rhoP[1]/2) + rhoP[0]

    return sam
    

def truePAC(params,piParams,eps,Lip,delta,a_p_c,m,alph,p,dim,arch,chi):
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
    rhoParams=[params[0],params[1]]
    piParams=[piParams[0],piParams[1]]
    NNsize=dimComp(arch)
    bb=2*(jnp.log(delta))/jnp.sqrt(m) + (.5*eps**2)/jnp.sqrt(m)
    kl=KLdiag_from_log_scale(piParams,rhoParams,NNsize)
    theta=(Lip*a_p_c + 1 )
    bb+= (1./jnp.sqrt(m))*kl + jnp.sqrt(theta/(2*m)*chi)
    return bb


def pacBound(params,piParams,eps,Lip,delta,a_p_c,a,m,alph,lambda_,p,dim,arch):
    '''
    Function that computes the value of the linearised Pac bound (3.14) in Curato et al. (2025)
    Input:
        params: parameters of the generalized distribution
        piParams: parameters of the reference distribution
        eps: truncation level of the loss function
        Lip: Lipschitz constant estimate of the network, computed over the reference distribution
        delta: probability level of the pav bound
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
    p=1
    bb=0
    rhoParams=[params[0],params[1]]#jnp.diag(params[1])]
    piParams=[piParams[0],piParams[1]]#jnp.diag(piParams[1])]
    NNsize=dimComp(arch)
    bb= 2*(jnp.log(delta))/jnp.sqrt(m) + (.5*eps**2)/jnp.sqrt(m)
    kl=KLdiag_from_log_scale(piParams,rhoParams,NNsize)
    theta=(Lip*a_p_c+1)  #lambda gamma: 
    bb+= (1./jnp.sqrt(m))*kl+jnp.sqrt((theta/m)*kl)
    return bb


def KLdiag_grad(piParams,rhoParams,NNsize):
    '''
    Function that computes the gradient of the Kullback-Leibler divergence between two multivariate diagonal gaussian distribution
    Input:
        piParams: parameters of the reference distribution
        rhoParams: parameters of the generalised posterior distribution
        NNsize: parameter dimension
    '''
    piParams0=piParams[0]
    piParams1=piParams[1]
    rhoParams0=rhoParams[0]
    rhoParams1=rhoParams[1]

    grad = lambda pip0,pip1,rhop0,rhop1: jnp.array([(rhop0-pip0)/pip1,.5*(1./pip1-1./rhop1)]) #(rhop0-pip0)/pip1
    gradient= jax.vmap(grad)(piParams0,piParams1,rhoParams0,rhoParams1)
    return jnp.transpose(gradient)

def pac_gradient(piParams,rhoParams,NNsize,m,Lip,a_p_c):
    '''
    Function that computes the gradient of the second term of the linearised pac bound
    Input:
        piParams: parameters of the reference distribution
        rhoParams: parameters of the generalised posterior distribution
        NNsize: parameter dimension
        m: number of cones of the embedding
        Lip: Lipschitz constant estimate of the network, computed over the reference distribution
        a_p_c: dimension of the network input         
    '''
    
    kl_grad= KLdiag_grad(piParams,rhoParams,NNsize)
    return kl_grad*(1/jnp.sqrt(m) + .5*jnp.sqrt((Lip*a_p_c+1)/(m*KLdiag_from_log_scale(piParams,rhoParams,NNsize))))

def pac_approx(piParams,rhoParams,NNsize,m,Lip,a_p_c):
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
    return KLdiag_from_log_scale(piParams,rhoParams,NNsize)*1./jnp.sqrt(m) + jnp.sqrt((Lip*a_p_c+1)*KLdiag_from_log_scale(piParams,rhoParams,NNsize)/m)


def pac_mc_allester(piParams,rhoParams,NNsize,m,Lip,a_p_c,delta):
    #preliminary version
    #mc allester-like bound 
    return jnp.sqrt( (KLdiag_from_log_scale(piParams,rhoParams,NNsize) + jnp.log(m*delta)) /(2*(m-1))  ) + jnp.sqrt((Lip*a_p_c+1)*KLdiag_from_log_scale(piParams,rhoParams,NNsize)/m)

def chi2_diag_gaussians(piParams,rhoParams):
    """
    Function that computes the chi-squared divergence between two gaussian distributions
    Input:
      piParams: parameters of the reference distribution
      rhoParams: parameters of the generalised posterior distribution
    """
    mu_p = rhoParams[0]
    var_p = rhoParams[1]
    mu_q = piParams[0]
    var_q = piParams[1]
    mu_p = jnp.array(mu_p)
    mu_q = jnp.array(mu_q)
    var_p = jnp.array(var_p)
    var_q = jnp.array(var_q)
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
    piParams: parameters of the reference distribution
    rhoParams: parameters of the generalised posterior distribution
      
    Returns
    kl: computation of the divergence
    -------
    
    '''
    
    piParams0=piParams[0]
    piParams1=piParams[1]
    rhoParams0=rhoParams[0]
    rhoParams1=rhoParams[1]
    inv=lambda beta: 1./beta
    kl=jnp.dot(jax.vmap(inv)(piParams1),rhoParams1) 
    kl= kl - NNsize
    diff=piParams0-rhoParams0
    prod=lambda a,b: a*b
    kl= kl + jnp.dot(diff,jax.vmap(prod)(jax.vmap(inv)(piParams1),diff)) 
    kl=kl + jnp.sum(jax.vmap(jnp.log)(piParams1)) 
    kl=kl - jnp.sum(jax.vmap(jnp.log)(rhoParams1)) 
    return kl
    

def KLdiag_from_log_scale(piParams,rhoParams,NNsize):
    
    '''
    computes the KL divergence for two multivariate gaussians
    

    Parameters
    ----------
    piParams: parameters of the reference distribution
    rhoParams: parameters of the generalised posterior distribution
    NNsize: parameters dimension
    Returns
    kl: computation of the divergence
    -------
    None.
    '''
    piParams0=piParams[0] 
    piParams1=piParams[1] 
    rhoParams0=rhoParams[0] 
    rhoParams1=rhoParams[1]
    inv=lambda beta: 1./beta
    kl= jnp.sum(jnp.exp(rhoParams1-piParams1)-1)
    diff=piParams0-rhoParams0
    prod=lambda a,b: a*b
    kl= kl + jnp.dot(diff,jax.vmap(prod)(jnp.exp(-piParams1),diff)) #matmul
    kl=kl + jnp.sum(piParams1) 
    kl=kl - jnp.sum(rhoParams1) 
    return kl
    



def LipC(piParams,dim,mask,arch,shard_size=int(5e2),N=int(1e3)):

    '''
    Computes the estimation of the Lipchitz constant for a ffnn
    Input:
        piParams: parameters of the reference distribution
        dim: network size
        mask: network layers structure
    Output L
    '''

  
    realizations = dist_sample(piParams,num_realizations=N,seed=jax.random.key(0))
    realizations=realizations.reshape((N//shard_size,shard_size,dim))
  
    def pozzo(params,mask,arch):
     
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo

    parPozzo=lambda alpha: pozzo(alpha,mask,arch)
    def naiveLip(weights):
      prod=1
      for el in weights:
        prod*=jnp.amax(jnp.abs(el))           #jnp.linalg.norm(el,ord=2)
      return prod
    L=0
    for el in realizations:
      w=jax.vmap(parPozzo)(el)

      c=jax.vmap(naiveLip)(w)
      L+= jnp.sum(c)
      
    L=L/N
    return L
  
  
def rescalingInv(d,slope,q,eps=0):
  
  return (d - q)/slope



def error(it_params,A_c_e,b_c_e,dim,arch,mask,piParams,eps,Lip,delta,a_p_c,a,m,alpha,lambda_,p,rng,shard_size=int(5e2),N=int(1e3)):

    '''
    Function computing the pac bound for a generic dataset, for a single spatial coordinate
    Outputs the two terms separately
    '''

    #lambda function, computes the value for the empirical risk
    MC_train_e = r_empirical_risk(A_c_e,b_c_e,arch, mask, dim, eps)

    # N draws from the predictive distribution
    realizations = dist_sample([it_params[0],it_params[1]],num_realizations=N,seed=rng)
    realizations = realizations.reshape((N//shard_size,shard_size,dim))
    error=0

    for el in realizations:
    #serial parallelization for each shard
      e = MC_train_e(el)
      error += jnp.sum(e)
    error=error/N
    pac=pacBound(it_params,piParams,eps,Lip,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch)
    return [error,pac]



def coordit_pretraining(Z,preT_coord,preT_d_slicing,a,c,p,arch,mask,dim,eps,init_scale=0.0016,pretraining_learning_rate=0.01):

  '''
  Function performing the pretraining stagel, over a predefined training subset, for a spatial coordinate
  Outputs the vector of parameters w, the random initialization w0, the emprical risk at pretraining
  '''

  #spatial slicing of the pretraining dataset
  preT_mapped=jax.vmap(preT_d_slicing)(preT_coord)
  preT_mapped=jnp.reshape(preT_mapped,(len(preT_coord),preT_mapped.shape[-1]))

  #cone extraction
  Ac_preT,bc_preT = Z.get_coneJ(preT_mapped,sizeData=preT_mapped.shape[1],preT=True)

  #parameter initialization
  init_log_scale=jnp.log(init_scale)
  initParams = [jnp.zeros(dim),jnp.ones(dim)*init_log_scale] 

  #random draw from the initial distribution
  w0 = dist_sample(initParams,seed=jax.random.key(0))
  w=w0

  #choice of optimizer, Adam by default
  optimizer = optax.adam(pretraining_learning_rate)	 
  opt_state = optimizer.init(w)

  #the number of epochs is intended to correspond to 12k iterations, in accordance to Dziugate, Roy (2017)
  for e in range(12):
    for i in range(Z.Ncones):
      #classic optimization routine
      scorf = l_empirical_risk(Ac_preT[:,i].reshape((Ac_preT[:,i].shape[0],1)),bc_preT[i],arch, mask, dim, eps)
      grads =jax.vmap(jax.grad(scorf))(w)
      updates, opt_state = optimizer.update(grads, opt_state)
      w = optax.apply_updates(w, updates)
  preT_val=jax.vmap(scorf)(w)
  return w,w0,preT_val

def makeh5(net,hdata,names):

  '''
  Takes as input the h5 file pointer, data to be stored and the respective names
  Creates and stores data structures in an (already defined!) h5 group
  '''
  for j in range(len(names)):
    net.create_dataset(names[j],data=hdata[j])



#@jit
def Optimization(file_m,Z,x_size,preT,rescaling,eps,delta,data,inp,p,c,arch,dim,Ndraws,Ncones,Ncones_test,Ncoords,shard_size,lr,slope=1,q=0,epochs=1,maxit=jnp.inf,piScaling=1):

    '''
    Performs the MMAF optimization routine 
    '''
  
    struct=[inp,*arch]
     
    print('number of epochs: ', epochs)


    #computes the parameter masking of the architecture
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
    

    if rescaling:
      print('rescaling between [-1,1]')
      data,pinv,qinv=rescalingU1(data)
      file_m.create_dataset('slope',data=pinv)
      file_m.create_dataset('quota',data=qinv)


    #train,validation,test split
    data_test=data[:,-(Ncones_test-1)*Z.a:] 
    data_train=data[:,:-Z.a*Ncones_test]
    data_pretraining = data_train[:,-1001*Z.a:] # extra cone for the last point 
    
    #adds the training dataset to the h5
    file_m.create_dataset('data_test',data=data_test)
  
    N=data_train.shape[1]
    x_size=data_train.shape[0]

    #creates the sharded mapping for secure parallelization
    list_windows= jnp.array([jnp.arange(element-c*p,element+c*p+1) for element in range(c*p,x_size-c*p)])
    
  
    preT_d_slicing= lambda beta: lax.dynamic_index_in_dim(data_pretraining,beta,axis=0) 
    

    # PRETRAINING stage
    pretraining_mean= jnp.zeros(dim)
    if preT:
      print('pretraining')
      pretraining = jax.vmap(lambda preT_coord: coordit_pretraining(Z,preT_coord,preT_d_slicing,Z.a,Z.c_,Z.p,[inp,*arch],mask,dim,eps))
      
      pretraining_mean,w0,preT_emp_risk = pretraining(list_windows) 
      print('saving preT_ER')
      file_m.create_dataset('pretraining ER',data=preT_emp_risk)

    ### PAC-BAYES BOUND OPTIMIZATION

  
    #data structure initialization
    coord = []
    rng = jax.random.key(0)
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


    #network parameters initialization
    if preT:
      #if pretraining is performed, the posterior mean is set as the resulting parameters of pretraining 
      
      pretraining_fun = jax.vmap(lambda beta: jnp.vstack([beta,jnp.ones(dim)*jnp.linalg.norm(beta,ord=1)]))
      pretraining_parameters = pretraining_fun(pretraining_mean)
      params=[pretraining_parameters[el*shard_size:(el+1)*shard_size] for el in range((x_size-2*p*c)//shard_size) ]
      piParams = [w0[0,0,:],piScale]
      
    else:
      #if no pretraining is performed, the posterior mean is set as having variance equal to 1/4
      rhoParams=[jnp.zeros(dim),jnp.ones(dim)*jnp.log(.25)] #jnp.ones(dim)/2
      sharded_params=jax.vmap(lambda dummy: jnp.vstack(rhoParams))(range(shard_size))
      params= [sharded_params for el in range((x_size-2*p*c)//shard_size)]   
      piParams=[jnp.zeros(dim),piScale]
      
   
    start_learning_rate = jnp.float32(lr)
  
    #computation of the Lipschitz function
    print('computing Lipschitz constant... ')
    Lip= LipC(piParams,dim,mask,[inp,*arch])
    print(Lip)


    #choice of OPTIMIZER, adam is selected by default

    optimizer = optax.adam(start_learning_rate)	
    #initial optimization stage, set accordingly to choice of pretraining
    if preT:
      opt_state_grid= [jax.vmap(optimizer.init)(el) for el in params]
    else:
      opt_state_grid = [jax.vmap(optimizer.init)(sharded_params) for el in range((x_size-2*p*c)//shard_size)]

    Acones_dummy = jnp.zeros((Ncones,inp))
    bcones_dummy = jnp.zeros(Ncones)
    
    sharded_Acones = jax.vmap(lambda dummy: jnp.vstack(Acones_dummy))(range(shard_size))
    Acones_s = [sharded_Acones for el in range((x_size-2*p*c)//shard_size)] 

    sharded_bcones = jax.vmap(lambda dummy: jnp.vstack(bcones_dummy))(range(shard_size))
    bcones_s = [sharded_bcones for el in range((x_size-2*p*c)//shard_size)] 

    
    time_windows=([jnp.arange(jnp.maximum(0,Z.N-Z.a*Z.Ncones*(element+1)),(Z.N-Z.a*Z.Ncones*element)) for element in range(Z.Nbatches)])
    t_slicing= lambda beta: lax.dynamic_index_in_dim(jnp.transpose(data_train), beta,axis=0)
    list_shards= ([jnp.array([jnp.arange(c_coord-p*c,c_coord+p*c+1) for c_coord in range(element,element+shard_size)]) for element in range(p*c,x_size-p*c,shard_size)])

    def ef_setup(it_coord,params,a,inp,arch,mask,Ndraws,Ncones_test,rng):
  
        '''
        Function that extracts the cones from the test set
        '''
      
        test_mapped=jax.lax.map(test_slicing,(it_coord))
        test_mapped=jnp.reshape(test_mapped,(len(it_coord),a*Ncones_test))
        Ac,bc=Z.get_coneJ((test_mapped),sizeData=test_mapped.shape[1])
        
        return [Ac,bc] 

    def coordit(it_coord,it_params,opt_state,batch,Bsize,m,dim,Lip,inp,rng=rng):

        '''
        Core function for the optimization: performs the gradient step and updates the parameters for each spatial coordinate at the current batch iteration
        '''

        #spatial training set slicing 
        window_mapped=jax.vmap(d_slicing)(it_coord)
        window_mapped=jnp.reshape(window_mapped,(len(it_coord),Bsize))

        #cone extraction for the current batch
        Acones,bcones=Z.get_coneJ((window_mapped),sizeData=time_windows[batch].shape[0])


        #lambda function, computes the empirical risk for a single draw from the posterior distribution
        scorf=l_empirical_risk(Acones,bcones[:],([inp,*arch]),mask,dim,eps)

        #computes the second term of the obj function (containing the divergence terms)
        pac_mapped = lambda beta: pac_approx(piParams,beta,dim,Ncones,Lip,inp)
        #pac_mapped= lambda beta : pac_mc_allester(piParams,beta,dim,m,Lip,inp,delta)

        #stochastic gradient estimator for the gradient of the expected empirical risk
        val_jest,jest = sge_pwj(scorf,it_params,my_multi_normal,rng)
              
        #computes the value and the gradient (using jax.grad) of the second term of the objective function
        val_grad=pac_mapped(it_params)
        grad= jax.grad(pac_mapped)(it_params)

      
        #updates the parameter via Adam update rule      
        updates = jest+grad
        updates, opt_state = optimizer.update(updates,opt_state,it_params)  
        it_params = optax.apply_updates(it_params,updates)
        
        return [it_params,opt_state,Acones,bcones,val_jest,val_grad]

    
    min_it=0 #will store the epoch index containing the best performing parameters
    min_error=jnp.inf
    milestone_seeds=[]

    #initialization of data structures containing optimization history
    val_jest_epoch= jnp.empty((Z.Nbatches,x_size-2*Z.c_*Z.p))
    val_grad_epoch= jnp.empty((Z.Nbatches,x_size-2*Z.c_*Z.p))

    
    for epoch in trange(1,epochs+1, desc='epochs', colour='green'):

        #creates a new group in the h5 file for the current epoch
        file_epoch= file_m.create_group('epoch'+str(epoch))
      
        val_jest_batch= jnp.empty((0,x_size-2*Z.c_*Z.p))
        val_grad_batch= jnp.empty((0,x_size-2*Z.c_*Z.p))
      
        for batch in range(Z.Nbatches):

            #random PRNG key initialization, ensures reproducibility 
            key = jax.random.PRNGKey(batch+Z.Nbatches*(epoch-1))
            #key mapping, to ensure safe parallelization
            keys = [jax.random.split(key*(el+1), shard_size).reshape(shard_size, 2) for el in range((x_size-2*p*c)//shard_size)]
            
            #batch slicing
            time_mapped=jax.vmap(t_slicing)(time_windows[batch])
            time_mapped=jnp.squeeze(time_mapped,axis=1)     
            
            shard_slicing = lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)
            d_slicing = lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)  #  [,:]

            #lambda function, parallelizes the optimization routine to each coordinates shard
            opt_mapping = jax.vmap(lambda og,pmap,opt,rngM: coordit(og,pmap,opt,batch,Z.Ncones*Z.a,Z.Ncones,dim,Lip,inp,rngM),in_axes=(0,0,0,0))
    
            output = jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(Ndraws)))
            
            val_jest_stacked= jnp.array([])
            val_grad_stacked= jnp.array([])
            params_stacked =jnp.transpose(jax.vmap(lambda dummy:jnp.array([[],[]]))(range(dim)))

            for ls_i in range((x_size-2*p*c)//shard_size):

              #parallelization is performed serially for each shard, to prevent memory fill
              
              [params[ls_i],opt_state_grid[ls_i],Acones_s[ls_i],bcones_s[ls_i],val_jest,val_grad]= opt_mapping(list_shards[ls_i], params[ls_i], opt_state_grid[ls_i], keys[ls_i])#jax.jit
          
        #mockup of tape_grad    
              params_stacked=jnp.vstack([params_stacked,params[ls_i]])
              val_jest_stacked=jnp.hstack([val_jest_stacked,val_jest])
              val_grad_stacked=jnp.hstack([val_grad_stacked,val_grad])

            val_jest_batch= jnp.vstack([val_jest_batch,val_jest_stacked])
            val_grad_batch= jnp.vstack([val_grad_batch,val_grad_stacked])
      
        val_jest_epoch=jnp.vstack([val_jest_epoch,val_jest_batch])
        val_grad_epoch=jnp.vstack([val_grad_epoch,val_grad_batch])
        

      
        # assuming each epoch is a milestone epoch, we store the value of the objective function, as well as the pac bound 
        milestone_train_error = jnp.array([])
        b_c_e_stacked=jnp.empty((0,Ncones_test-1))
        test_error = jnp.array([])
        bound_train = jnp.array([])
        bound_test = jnp.array([])
      
        milestone_seeds.append(jax.random.key(epoch))

        test_slicing = lambda beta: lax.dynamic_index_in_dim(data_test,beta,axis=0)
        
        #lambda function extracting test cones from the test set
        finalStep=lambda coord,beta: ef_setup(coord, beta,Z.a,inp, arch,mask,Ndraws,Ncones_test-1,milestone_seeds[-1])

        #lambda function, computes the in-sample linearised pac bound (for the training set)
        e_mapped_train= jax.vmap(lambda  it_pars,it_A,it_b: error(it_pars,it_A,it_b,dim,[inp,*arch],mask,piParams,eps,Lip,delta,inp,Z.a,Z.Ncones,Z.alpha,Z.lambda_,Z.p,milestone_seeds[-1]))
        
        #lambda function, computes the out-of-sample linearised pac bound (for the test set)
        e_mapped_test = jax.vmap(lambda  it_pars,it_A,it_b: error(it_pars,it_A,it_b,dim,[inp,*arch],mask,piParams,eps,Lip,delta,inp,Z.a,Ncones_test-1,Z.alpha,Z.lambda_,Z.p,milestone_seeds[-1]))

        for ls_i in range((x_size-2*p*c)//shard_size): 
          #stores the pac bound values accordingly
              mile_train_e,pac_train = e_mapped_train(params[ls_i],Acones_s[ls_i],bcones_s[ls_i])
              milestone_train_error=jnp.hstack([milestone_train_error,mile_train_e])
              bound_train = jnp.hstack([bound_train,pac_train])

          #extraction of the test set cones
              A_c_e,b_c_e=jax.vmap(finalStep)(list_shards[ls_i],params[ls_i]) #save b_c_e_stacked
              b_c_e_stacked=jnp.vstack([b_c_e_stacked,b_c_e])
          #computation of the out-of-sample pac bound
              test_e,pac_test= e_mapped_test(params[ls_i],A_c_e,b_c_e)
              test_error=jnp.hstack([test_error,test_e])   
              bound_test = jnp.hstack([bound_test,pac_test])  

        if jnp.mean(bound_train+milestone_train_error,axis=0)<min_error: 
        #updates the best parameters                
          min_it=epoch
          min_error=jnp.mean(bound_train+milestone_train_error,axis=0)
          best_params=params_stacked

      #stores in the current epoch group all the measures for future diagnostic 
        hdata=[Z.a,Z.lambda_, Z.Ncones,bound_train,milestone_train_error,test_error,bound_test,val_jest_batch,val_grad_batch,b_c_e_stacked]
        names=[ 'a','lambda', 'm', 'bound_train', 'train_errors','test_errors','bound_test','val_jest','val_grad','cones_test']
        makeh5(file_epoch,hdata,names)

    file_last=file_m.create_dataset('last params',data=params_stacked)

    #creates the h5 group for the best parameters
    file_best=file_m.create_group('min')
    file_best.create_dataset('min_error',data=min_error)
    file_best.create_dataset('min iteration',data=min_it)
    file_best.create_dataset('best params',data=best_params)
        
