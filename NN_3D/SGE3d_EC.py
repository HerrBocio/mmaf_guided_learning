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

from scipy.stats import randint
from tqdm import trange,tqdm
#from optax._src import wrappers,utils
#from optax.monte_carlo import stochastic_gradient_estimators as sge
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'#,2,3'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

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

  

def get_simulated_data(filename):
    data=loadmat(filename+'.mat')
    data=data["data"]

    #data=data-np.mean(data)
    #data=data/np.std(data)

    return data

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

    return lambda beta: empirical_risk(A,b,beta,arch,mask,dim,eps)

def r_empirical_risk(A,b,arch,mask,dim,eps):

    return lambda beta: empirical_val(A,b,beta,arch,mask,dim,eps)


def empirical_risk(A,b,realization,arch,mask,dim,eps,num_realizations=1):
    
    #print(realization.shape)          
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo

    #print(mask,type(mask))
    #print(arch)
    realization=pozzo(realization,mask,arch)
    
    #print(realization[-1].shape)
    #print('iner')
    #print('realShape',realization.shape)
    empR=lambda beta : get_loss_function(A,b,beta,eps)
    #realization=post(rhoParams)#,num_realizations=20)
    #print(realization.shape)
    eU=empR(realization)
    
    #c=np.mean(eU)
    #print("n_c: ",c).
    #print(type(eU))
    return eU

def empirical_val(A,b,realization,arch,mask,dim,eps,num_realizations=10):
    
    #print(realization.shape)          
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo
    masking= lambda a: pozzo(a,mask,arch)
  
    realization=jax.vmap(masking)(realization)
    empR=lambda beta : return_loss_function(A,b,beta,eps)
    #realization=post(rhoParams)#,num_realizations=20)
    #print(realization.shape)
    eU=empR(realization)
    return eU



#does it make sense to differentiate prior and posterior, when both are gaussian

def get_loss_function(A,b,weights,eps=2.99):


    #r_eps=0
    #print(arch)
    #print(A.shape)
    #fun_sq = lambda beta: ffnnLossJ(A,arch,beta,b) 

    fun_map = lambda beta: ffnnLossPozzo(A,beta,b,eps) 

    
    r_eps= fun_map(weights)
  
    return r_eps
    
 
def return_loss_function(A,b,weights,eps=2.99):

    #r_eps=0

    #print(weights.shape)
    #fun_sq = lambda beta: ffnnLoss(A,arch,beta,b) 

    fun_map = lambda beta: ffnnLossPozzo(A,beta,b,eps) 

    
    r_eps= jax.vmap(fun_map)(weights)
  
    #jax.lax.map(f, xs, *, batch_size=None)
    return r_eps	#weights.shape keeps track of the sample size of the monte carlo estimator

def makeh5(net,hdata,names):
	
  #for i in range(len(datasets)):
  #x=net.create_group(str(dataset))
  for j in range(len(names)):
    net.create_dataset(names[j],data=hdata[j])
  #file_.close()




def dimComp(archs):
	
	dim=0
	#dim=inp*(archs[0]+1)
	for i in range(len(archs)-1):
		dim = dim + (archs[i]*(archs[i+1])+archs[i+1])
	return dim



def post(rhoP,num_realizations=1,seed=1):
    
    '''
    The parameters of the neural network are stored in a linear vector, to prevent memory fill.
    
    '''
    sample_shape = (num_realizations,*rhoP[0].shape)#tuple(num_realizations) + rhoP[0].shape jax.random.key(seed)
    sam=jax.random.normal(seed, shape=sample_shape) * rhoP[1] + rhoP[0]
    #print(rhoP)
    #print('size' ,size)
    #x=utils.multi_normal.sample(
    #x=jax.random.multivariate_normal(jax.random.key(seed),rhoP[0],rhoP[1],(num_realizations,))#,method='svd')#,size)#
    #print('x: ',x)#.shape)
    #x=jnp.reshape(x,(num_realizations,*size))
    #print(x.shape)
    #print('\n\n\n\n shape',x.shape,'\n\n\n\n\n')
    return sam
    
def prior(piP,num_realizations=1,seed=0):
    
    
    sample_shape = (num_realizations,*piP[0].shape)#tuple(num_realizations) + rhoP[0].shape
    sam=jax.random.normal(jax.random.key(seed), shape=sample_shape) * piP[1] + piP[0]
    
    #print(size)
    #print
    #x=jax.random.multivariate_normal(random.key(seed),piP[0],piP[1],(num_realizations,))	#seeded!!!
    #x=jnp.reshape(x,(num_realizations,*size))
    #print('prior shape',x.shape)
    return sam	


def pacA(piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch):

    return lambda beta: pacApprox(beta,piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch)


def pac_approx(piParams,rhoParams,NNsize,m,Lip,a_p_c):

    return KLdiag_from_log_scale(piParams,rhoParams,NNsize)*1./jnp.sqrt(m) + jnp.sqrt((Lip*a_p_c+1)*KLdiag_from_log_scale(piParams,rhoParams,NNsize)/m)




def pacBound(params,piParams,eps,Lip,delta,a_p_c,a,m,alph,lambda_,p,dim,arch):
    p=1
    bb=0
    rhoParams=[params[0],params[1]]#jnp.diag(params[1])]
    piParams=[piParams[0],piParams[1]]#jnp.diag(piParams[1])]
    NNsize=dimComp(arch)
    bb= 2*(jnp.log(delta))/jnp.sqrt(m) + (.5*eps**2)/jnp.sqrt(m)
    kl=KLdiag_from_log_scale(piParams,rhoParams,NNsize)
    theta=(Lip*a_p_c+1)  #lambda gamma: 
    bb= (1./jnp.sqrt(m))*kl+jnp.sqrt((theta/m)*kl)
    return bb

def pacBoundE(params,piParams,eps,delta,a_p_c,a,m,alph,lambda_,p,dim,arch):
    p=1
    bb=0
    rhoParams=[params[0],params[1]]#jnp.diag(params[1])]
    piParams=[piParams[0],piParams[1]]#jnp.diag(piParams[1])]
    NNsize=dimComp(arch)
    bb= 2*(jnp.log(delta))/jnp.sqrt(m) + (.5*eps**2)/jnp.sqrt(m)
    kl=KLdiag(piParams,rhoParams,NNsize)
    theta=(naiveLip(prior(piParams),arch)*a_p_c+1)*alph*jnp.exp(-lambda_*(a-p))  #lambda gamma: 
    bb= (1./jnp.sqrt(m))*kl+jnp.sqrt(((eps*delta*theta/m)*2*kl))
    return bb
  
def KLdiag(piParams,rhoParams,NNsize):
    
    '''
    computes the KL divergence for two multivariate gaussians
    

    Parameters
    ----------
    piP : parameters of prior distribution.
    rhoP : parameters of posterior distribution.

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
    #print('kl end',kl)
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
    


def LipC(piParams,dim,mask,arch,shard_size=int(5e2),N=int(1e3)):

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

   

def get_loss_vector(A,b,w,arch,loss,eps=2.99):
    
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



def coordit_pretraining(Z,preT_coord,data_preT,a,c,p,arch,mask,dim,eps,init_scale=0.0016,pretraining_learning_rate=0.01):

  # assuming the parallelization already took place
  #preT_mapped=jax.vmap(preT_d_slicing)(preT_coord)
  def mesh(ind):
    m,n=jnp.meshgrid(ind[0],ind[1],indexing='ij')
    return [m,n]
  preT_mapped=data_preT[:,*mesh(preT_coord)]#print('coord',x_coord)
  #preT_mapped=jnp.reshape(preT_mapped,(len(preT_coord),preT_mapped.shape[-1]))
  print(preT_mapped.shape)
  
  Ac_preT,bc_preT = Z.get_coneJ_3d(preT_mapped,sizeData=preT_mapped.shape[1],preT=True)

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
  for e in range(5):
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




def error(it_params,A_c_e,b_c_e,dim,arch,mask,piParams,eps,Lip,delta,a_p_c,a,m,alpha,lambda_,p,rng,shard_size=int(5e2),N=int(1e3)):


    MC_train_e = r_empirical_risk(jnp.transpose(A_c_e),b_c_e,arch, mask, dim, eps)
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



#@jit
def Optimization(file_m,Z,preT,rescaling,eps,delta,data,inp,p,c,arch,dim,Ndraws,Ncones,Ncones_test,Ncoords,shard_size,lr,rhoScaling,slope=1,q=0,epochs=1,maxit=jnp.inf,piScaling=1,acrit=.25):
    
    struct=[inp,*arch]
  
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
       #print(mask)   	


    if rescaling:
      print('rescaling between [-1,1]')
      data,pinv,qinv=rescalingU1(data)             # TO DO
    print('minmax',np.amin(data),np.amax(data))
    data_test=data[-(Ncones_test-1)*Z.a:,:,:] 
    print('data test',data_test.shape)
    data_train=data[:-Z.a*Ncones_test,:,:]
    data_pretraining = data_train[-1001*Z.a:,:,:] # extra cone for the last point 
    #print(data_pretraining.shape)
    file_m.create_dataset('data_test',data=data_test)
  
    N=data.shape[0]
    x1_size=data_train.shape[1]
    x2_size=data_train.shape[2]
    #x_size=last_cone.shape[0]
    #print('in sgd',x_size)
    #time.sleep(10)
	#trange(Ndraws, position=0, desc="r", leave=True, colour='green'):rrrrrrr
    rng = jax.random.key(0)
    
    #list_windows_x1= jnp.array([jnp.arange(element-p,element+p+1) for element in range(p,x1_size-p)])
    #list_windows_x2= jnp.array([jnp.arange(element-p,element+p+1) for element in range(p,x2_size-p)])

    #preT_d_slicing= lambda beta: lax.dynamic_index_in_dim(data_pretraining,beta,axis=0) 
    list_pretraining = jnp.array(
                  [
                   jnp.array([jnp.arange(c_coord_x1-2*Z.p*Z.c_,c_coord_x1+2*Z.p*Z.c_+1),jnp.arange(c_coord_x2-2*Z.p*Z.c_,c_coord_x2+2*Z.p*Z.c_+1)]) 
                   for c_coord_x1 in range(p*c,x1_size-p*c,shard_size)   
                   for c_coord_x2 in range(p*c,x2_size-p*c,shard_size) 
                  ]
                  )

    # TO DO WITH c,p != 1
  
    list_shards= [jnp.array(
                  [
                   jnp.array([jnp.arange(c_coord_x1-2*Z.p*Z.c_,c_coord_x1+2*Z.p*Z.c_+1),jnp.arange(c_coord_x2-2*Z.p*Z.c_,c_coord_x2+2*Z.p*Z.c_+1)]) 
                   for c_coord_x1 in range(element1,element1+shard_size)   
                   for c_coord_x2 in range(element2,element2+shard_size) 
                  ]
                  )
                   for element1 in range(p*c,x1_size-p*c,shard_size) for element2 in range(p*c,x2_size-p*c,shard_size)
                 ]
    #print('qui ',Z.Nbatches)
    
    pretraining_mean= jnp.zeros(dim)
    # PRETRAINING
    if preT:
      print('pretraining')
      pretraining = jax.vmap(lambda preT_coord: coordit_pretraining(Z,preT_coord,data_pretraining,Z.a,Z.c_,Z.p,[inp,*arch],mask,dim,eps))
      
      pretraining_mean,w0,preT_emp_risk = pretraining(list_pretraining) 
      print('saving preT_ER')
      file_m.create_dataset('pretraining ER',data=preT_emp_risk)
    #print('w0',w0[0,:,:].shape)
    # PAC-BAYES BOUND OPTIMIZATION
  
        
    
    coord=jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(3)))
    for t in reversed(range(p)):  # self.p = p
        # aggiunge alla fine di coord il vettore dentro tf.constant()
      coord1=jnp.array([jnp.array([- (t+1), v, u]) for v in range(-int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1), int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1) + 1)  for u in range(-int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1), int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1) + 1)])
      coord=jnp.vstack([coord,coord1])   
    
    bounds=[]#np.zeros(Nsteps)
    pars=[]
    #print(coord)  
    
    
    #rhoScale=jax.vmap(scaleInit,in_axes=(0,0))(aux_in,jnp.array(arch))
    #print(piScale)
    #rhoParams =[(jnp.ones(dim))*rhoScaling,jnp.ones((dim))*1.5]#]#jnp.array(loc),jnp.array(scale)


  
    if piScaling!=1:
      print('fixed setup')
      piScale=jnp.ones(dim)/piScaling
    else: print('mixed setup')

    if preT:
      
      pretraining_fun = jax.vmap(lambda beta: jnp.vstack([beta,jnp.ones(dim)*jnp.linalg.norm(beta,ord=1)]))
      pretraining_parameters = pretraining_fun(pretraining_mean)
      #pretraining_parameters = list(pretraining_parameters.reshape(((x_size-2*p*c)//shard_size,shard_size,2,dim))))
      params=[pretraining_parameters[el*shard_size:(el+1)*shard_size] for el in range((x1_size-2*p)//shard_size*(x2_size-2*p)//shard_size)  ]
      #print('preT_params',params)
      #pretraining_fun_prior = jax.vmap(lambda beta: jnp.vstack([beta,jnp.ones(dim)*piScale]))
      #piParams_preT = pretraining_fun_prior(w0)#SET BACK GRID SEARCH INIT!!!
      #piParams = [piParams_preT[el*shard_size:(el+1)*shard_size] for el in range((x_size-2*p*c)//shard_size) ]
      piParams = [w0[0,0,:],piScale]

    else: 
      piParams = [jnp.zeros(dim),piScale]
      rhoParams=[jnp.zeros(dim),jnp.ones(dim)*jnp.log(.25)] #jnp.ones(dim)/2
      sharded_params=jax.vmap(lambda dummy: jnp.vstack(rhoParams))(range(shard_size))
      params= [sharded_params for el in range((x1_size-2*p)//shard_size*(x2_size-2*p)//shard_size)]   
      #print(params)
  
    print('computing Lipschitz constant... ')
    Lip= LipC(piParams,dim,mask,[inp,*arch])
    print(Lip)
    #jax.clear_caches()

    Acones_dummy = jnp.zeros((Ncones,inp))
    bcones_dummy = jnp.zeros(Ncones)
    
    sharded_Acones = jax.vmap(lambda dummy: jnp.vstack(Acones_dummy))(range(shard_size))
    Acones_s = [sharded_Acones for el in range((x1_size-2*p*c)//shard_size*(x2_size-2*p*c)//shard_size)] 

    sharded_bcones = jax.vmap(lambda dummy: jnp.vstack(bcones_dummy))(range(shard_size))
    bcones_s = [sharded_bcones for el in range((x1_size-2*p*c)//shard_size*(x2_size-2*p*c)//shard_size)] 


  
    #sharded_params=jax.vmap(lambda dummy1: jnp.hstack(jax.vmap(lambda dummy2: jnp.vstack(rhoParams))(dummy1)))(jnp.arange(shard_size**2))
    sharded_params= jax.vmap(lambda dummy: jnp.vstack(rhoParams))(jnp.arange(shard_size**2))
    #print(sharded_params.shape)
    params= [sharded_params for el in range((x1_size-2*p)//shard_size*(x2_size-2*p)//shard_size)]##jnp.tile(jnp.hstack(rhoParams),((x_size-2*p),1)) # correct w/ cone modulation 
    #print('ps',params.shape)
    #print(jnp.vstack(rhoParams).shape,(x_size-2*p))  #jax.tree_util.tree_map(jnp.asarray,jnp.tile(jnp.vstack(rhoParams),x_size))#{'mean': rhoParams[0],'cov' : rhoParams[1] }
    #print('size ', (x_size-2*p-1)//shard_size)    
    start_learning_rate = jnp.float32(lr)
    
    #OPTIMIZER
    #opt_states=[]
    optimizer = optax.adam(start_learning_rate)	#SGD ON
    #opt_state=optimizer.init(jnp.vstack(rhoParams))
    #for el in range(x_size):
    #  opt_states.append(opt_state)
    # Vectorized optimizer state init
    #opt_state_grid = [jax.vmap(optimizer.init)(sharded_params) for el in range(((x1_size-2*p)//shard_size)*((x2_size-2*p)//shard_size))]
    #print(opt_state_grid[0])
    if preT:
      opt_state_grid= [jax.vmap(optimizer.init)(el) for el in params]
    else:
      opt_state_grid = [jax.vmap(optimizer.init)(sharded_params) for el in range(((x1_size-2*p*c)//shard_size)*((x2_size-2*p*c)//shard_size))]
       
    #last_cone_slicing= lambda beta: lax.dynamic_index_in_dim(last_cone, beta,axis=0)
    #for x_coord in range(Ncoords):#,40):#(x_size, position=0,leave=True, desc="coordinate", colour='red'):    
    #@partial(jit,static_argnums=1)
    #print(Z.Ncones,Z.a)
    time_windows=([jnp.arange(jnp.maximum(0,Z.N-Z.a*Z.Ncones*(element+1)),(Z.N-Z.a*Z.Ncones*element)) for element in range(Z.Nbatches)])
    #print('t w',time_windows)
    t_slicing= lambda beta: lax.dynamic_index_in_dim(data_train, beta,axis=0) 
    
    test_slicing= lambda beta: lax.dynamic_index_in_dim(data_test, beta,axis=0) 
    
    test_mapped=jax.vmap(test_slicing)(jnp.arange(0,data_test.shape[0]))#print('coord',x_coord)
    #print(test_mapped.shape)
    test_mapped=jnp.squeeze(test_mapped,axis=1)
    #print(test_mapped.shape)
            
    '''
    def ef(it_coord,data,params,piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch):

        window_mapped=jax.lax.map(d_slicing,(it_coord))#print('coord',x_coord)
        # print('shape',window_mapped.shape,batch)
        window_mapped=jnp.reshape(window_mapped,(len(it_coord),Z.Bsize))
        #print('shape',window_mapped.shape)
        
        #cone_mapped= jax.lax.map(last_cone_slicing,(it_coord))
        #print(cone_mapped.shape)
        #cone_mapped=jnp.reshape(cone_mapped,(len(it_coord),Z.a ))
        #print(params[0].shape)
        Ac,bc=Z.get_coneJ((window_mapped),sizeData=window_mapped.shape[1])
        scorfval=(r_empirical_risk( Ac,bc,arch,mask,dim,loss,eps,num_realizations=20))  
        key = jax.random.PRNGKey(batch)
        finalSge=scorfval(post([params[0],params[1]],num_realizations=10,seed=key))#jax.jit  	
        sm=jnp.mean(finalSge)   
      
        #print('pac',pacBound(params,piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch))
        
        finalBound= sm + pacBound(params,piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch)    
        return finalBound
    '''
    def multi_ef(it_coord,params,a,inp,arch,mask,Ndraws,Ncones_test,rng):

        def mesh(ind):
          m,n=jnp.meshgrid(ind[0],ind[1],indexing='ij')
          return [m,n]
        #print(mesh(it_coord))
        test_coord=test_mapped[:,*mesh(it_coord)]
        
        #print('shape',test_coord.shape)
        #test_coord=jnp.reshape(test_coord,a*Ncones_test),*it_coord
        #print(test_coord.shape[0])
        Ac,bc=Z.get_coneJ_3d(test_coord,sizeData=test_coord.shape[0])
        #print(Ac.shape,bc.shape)
        #w=post([params[0],params[1]],Ndraws,seed=rng)    
        #print(w.shape)
        #out=jax.vmap(lambda A : ffnnV(A,[inp,*arch],mask,w))
        #m_e_f = out(jnp.transpose(Ac))
        #m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        #print('mef',m_e_f.shape)
        
        return [Ac,bc]

  
    def coordit(it_coord,it_params,opt_state,batch,Bsize,slope,q,rng=rng):
        #if not x_coord%10: print('cood: ',x_coord)
        #x_coord=jnp.where(it_coord<Ncoords-p,it_coord,0)
        #print(it_coord)
        #Z.data=data
        #print(it_params.shape[0])
        #it_params=jnp.reshape(it_params,(2,int(it_params.shape[0]/2)))
        #print('in init coord')   ####################################################
        #print(it_coord)
        #print(rng)
      
        def mesh(ind):
          m,n=jnp.meshgrid(ind[0],ind[1],indexing='ij')
          return [m,n]
        #print(mesh(it_coord))
        window_mapped=time_mapped[:,*mesh(it_coord)]#print('coord',x_coord)
        #print('shape',window_mapped.shape)
        #window_mapped=jnp.reshape(window_mapped,(len(it_coord),Bsize))
        #print('shape',window_mapped.shape)
        #print(last_cone.shape)
        
        #cone_mapped=  last_cone[:,*mesh(it_coord)]#jax.lax.map(last_cone_slicing,(it_coord))
        #print(cone_mapped)
        #cone_mapped=jnp.reshape(cone_mapped,(len(it_coord),Z.a ))
        #print('shape',cone_mapped.shape)
        Acones,bcones=Z.get_coneJ_3d((window_mapped),sizeData=window_mapped.shape[0])#,sizeData=time_windows[batch].shape[0])
        input_size,Z.m=Acones.shape
        num_realizations=1
        #print(Acones.shape,bcones.shape)

      
        #fun=(pacG(Acones,bcones,piParams,eps,delta,Z.a,Z.m,Z.alpha,Z.lambda_,dim,jnp.array([inp,*arch]),loss))  # jax.jit
        #print(Acones.shape,bcones.shape)
        #A,b=Z.get_coneJ_3d(cone_mapped,sizeData=cone_mapped.shape[0])
        #funApprox=(pacA(piParams,eps,delta,A.shape[0],Z.a,Z.m,Z.alpha,Z.lambda_,Z.p,dim,([inp,*arch])))  # jax.jit
        sgds=[]#np.zeros((Nsteps,2,len(rhoParams[0])))
        sges=[]#np.zeros(Nsteps)												
        scorf=l_empirical_risk(jnp.transpose(Acones),jnp.transpose(bcones),([inp,*arch]),mask,dim,eps)#jax.jit
        #scorfval=r_empirical_risk(jnp.transpose(Acones),jnp.transpose(bcones),([inp,*arch]),mask,dim,eps)#jax.jit  	
        
        pac_mapped= lambda beta: pac_approx(piParams,beta,dim,Ncones,Lip,inp) 
  
  
        grad = jax.grad(pac_mapped)(it_params) #EXPERIMENTAL VERSION!!! CHANGE
        
        val_grad = pac_mapped(it_params)

        
        #print('grad',grads)
        val_jest,jest = sge_pwj(scorf,it_params,my_multi_normal,rng)# 10 is just for example, 

        updates = jest + grad 
        #print(updates)
        #print(opt_state)
        updates, opt_state = optimizer.update(updates,opt_state,it_params)#print(opt_state)
        #print('jest',jest)	
        it_params = optax.apply_updates(it_params,updates)
        #print(it_params)
        
        return [it_params,opt_state,Acones,bcones,val_jest,val_grad]




    countit=0
    #best_params=[]
    min_it=0
    min_error=jnp.inf
    milestone_seeds=[]
    val_jest_epoch= jnp.empty((Z.Nbatches,(x1_size-2*Z.c_*Z.p)*(x2_size-2*Z.c_*Z.p)))
    val_grad_epoch= jnp.empty((Z.Nbatches,(x1_size-2*Z.c_*Z.p)*(x2_size-2*Z.c_*Z.p)))

    #print('bsize',Z.c_-c,Z.p-p)

    for epoch in trange(epochs, desc='epochs', colour='blue'): #range(1,epochs+1):
        #print('EPOCH: ', epoch)
        file_epoch= file_m.create_group('epoch'+str(epoch))
        val_jest_batch= jnp.empty((0,(x1_size-2*Z.c_*Z.p)*(x2_size-2*Z.c_*Z.p)))
        val_grad_batch= jnp.empty((0,(x1_size-2*Z.c_*Z.p)*(x2_size-2*Z.c_*Z.p)))
          
        #print('NB',Z.Nbatches,x1_size-2*Z.c_*Z.p)
        for batch in range(Z.Nbatches):
        #def batching(batch)#,params=params,opt_state=opt_state,rng=rng):: int,len(range(0,x_size,shard_size))
        #here starts a new batch init
            #subkey,rng=jax.random.split(rng)
            key = jax.random.PRNGKey(batch)
            keys = [jnp.array(jax.random.split(key*(el+1), shard_size**2)) for el in range((((x1_size-2*p)//shard_size)*((x2_size-2*p)//shard_size)))]#print(rng,subkey)
            #print(type(keys[0]))#print((x1_size-2*p)//shard_size*(x2_size-2*p)//shard_size)
            time_mapped=jax.vmap(t_slicing)(time_windows[batch])#print('coord',x_coord)
            #print(time_mapped.shape)
            time_mapped=jnp.squeeze(time_mapped,axis=1) 
            last_cone=jax.vmap(t_slicing)(jnp.arange(N-Z.a,N,1))
            last_cone=jnp.squeeze(last_cone,axis=1) 
            #jnp.reshape(time_mapped,(len(it_coord),Z.Bsize),order='F')
            #shard_slicing=lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)
            #Z.data=time_mapped
            #print(time_mapped.shape)
            #print(list_shards)
            #d_slicing= lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)  #  [,:]
            #ccc=jax.vmap(lambda x1,pmap1,opt1,rngM1 : jax.vmap(lambda x2,pmap2,opt2,rngM2: coordit(x2,pmap,opt,batch,Z.Ncones*Z.a,slope,q,rngM),in_axes=(0,0,0,0))(x1,pmap1,opt1,rngM1),in_axes=(0,0,0,0))
            ccc=jax.vmap(lambda x,pmap,opt,rngM: coordit(x,pmap,opt,batch,Z.Ncones*Z.a,slope,q,rngM),in_axes=(0,0,0,0))
            #print(time_mapped[:,list_shards[0]]   )
            #window_sharded = jax.vmap(shard_slicing)(list_shards)
            #print(window_sharded.shape)
            output=jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(Ndraws)))

            val_jest_stacked= jnp.array([])
            val_grad_stacked= jnp.array([])
            params_stacked =jnp.transpose(jax.vmap(lambda dummy:jnp.array([[],[]]))(range(dim)))

          
            for ls_i in range(((x1_size-2*p*c)//shard_size)*((x2_size-2*p*c)//shard_size)):
              #print('ls_i',list_shards[ls_i],ls_i)#print(list_windows)z
              #print(opt_state_grid[ls_i])
              #print(type(opt_state_grid[0].mu))
              #slice_=jnp.arange(ls_i*shard_size,(ls_i+1)*shard_size)
              #opt_i=(opt_state_grid[0].count[slice_],opt_state_grid[0].mu[slice_],opt_state_grid[0].nu[slice_])
              [params[ls_i],opt_state_grid[ls_i],Acones_s[ls_i],bcones_s[ls_i],val_jest,val_grad]= ccc(list_shards[ls_i],params[ls_i],opt_state_grid[ls_i],keys[ls_i])#jax.jit
              #print('out of coordit')
              #updates, opt_states = jax.vmap(stateMap,zip(updates,opt_states))
              #pitdist= jnp.bincount(pit_vals,jnp.ones((x_size,))/x_size).cdf(low=0,high=51)
              params_stacked=jnp.vstack([params_stacked,params[ls_i]])
              val_jest_stacked=jnp.hstack([val_jest_stacked,val_jest])
              val_grad_stacked=jnp.hstack([val_grad_stacked,val_grad])

            val_jest_batch= jnp.vstack([val_jest_batch,val_jest_stacked])
            val_grad_batch= jnp.vstack([val_grad_batch,val_grad_stacked])

        val_jest_epoch=jnp.vstack([val_jest_epoch,val_jest_batch])
        val_grad_epoch=jnp.vstack([val_grad_epoch,val_grad_batch])

        #print(params_stacked.shape)
        milestone_train_error = jnp.array([])
        #multi_e_f_stacked=jnp.empty((0,Ndraws,Ncones_test-1))
        b_c_e_stacked=jnp.empty((0,Ncones_test-1))
        test_error = jnp.array([])
        bound_train = jnp.array([])
        bound_test = jnp.array([])
      
        milestone_seeds.append(jax.random.key(epoch))


        finalStep=lambda coord,beta: multi_ef(coord, beta,Z.a,inp, arch,mask,Ndraws,Ncones_test-1,milestone_seeds[-1])

        e_mapped_train= jax.vmap(lambda  it_pars,it_A,it_b: error(it_pars,it_A,it_b,dim,[inp,*arch],mask,piParams,eps,Lip,delta,inp,Z.a,Z.Ncones,Z.alpha,Z.lambda_,Z.p,milestone_seeds[-1]))
        e_mapped_test = jax.vmap(lambda  it_pars,it_A,it_b: error(it_pars,it_A,it_b,dim,[inp,*arch],mask,piParams,eps,Lip,delta,inp,Z.a,Ncones_test-1,Z.alpha,Z.lambda_,Z.p,milestone_seeds[-1]))

        for ls_i in range(((x1_size-2*p*c)//shard_size)*((x2_size-2*p*c)//shard_size)):
      
          mile_train_e,pac_train = e_mapped_train(params[ls_i],Acones_s[ls_i],bcones_s[ls_i])
          milestone_train_error=jnp.hstack([milestone_train_error,mile_train_e])
          bound_train = jnp.hstack([bound_train,pac_train])
          #print('b train', jnp.mean(bound_train))
          A_c_e,b_c_e=jax.vmap(finalStep)(list_shards[ls_i],params[ls_i]) #save b_c_e_stacked
          #multi_e_f_stacked=jnp.vstack([multi_e_f_stacked,multi_e_f])
          #print(b_c_e.shape)
          b_c_e_stacked=jnp.vstack([b_c_e_stacked,b_c_e])
          test_e,pac_test= e_mapped_test(params[ls_i],A_c_e,b_c_e)
          test_error=jnp.hstack([test_error,test_e])   
          bound_test = jnp.hstack([bound_test,pac_test])  

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
  
    