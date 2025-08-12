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
from STOUpozo import *
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


class MyMultiNormalDiagFromLogScale:	
#to be placed in a specific scritp (collect other distros?)
#rewrite class wrt to functional syntax
  """MultiNormalDiag which directly exposes its input parameters."""

  def __init__(self, loc, scale,seed):
    self._scale = scale
    self._log_scale = jnp.log(scale)
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
    sam=jax.vmap(lambda k : jax.random.normal(k, shape=sample_shape) * (self._scale) + (self._mean) )(subkeys)
    #sam=jax.random.multivariate_normal(seed,self._mean,self._scale, size)lax.stop_gradient
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
  return MyMultiNormalDiagFromLogScale(loc=params[0],scale=jnp.sqrt(params[1]),seed=key)#, scale=jnp.diag(params[1]))

def sge_pwj(function,params,dist_builder,rng,num_samples):
  #subkeys=jax.random.split(self.seed,num=num_samples)
  #print(subkeys)
  def surrogate(params):
      # We vmap the function application over samples - this ensures that the
      # function we use does not have to be vectorized itself.
      dist = dist_builder(rng,*params)
      return jax.vmap(function)(dist.sample((num_samples,)))
  
  return jax.jacfwd(surrogate)(params)


def l_empirical_risk(A,b,arch,mask,dim,loss,eps,num_realizations=1):

    return lambda beta: empirical_risk(A,b,beta,arch,mask,dim,loss,eps,num_realizations)

def r_empirical_risk(A,b,arch,mask,dim,loss,eps,num_realizations=1):

    return lambda beta: empirical_val(A,b,beta,arch,mask,dim,loss,eps,num_realizations)


def empirical_risk(A,b,realization,arch,mask,dim,loss,eps,num_realizations=1):
    
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
    empR=lambda beta : get_loss_function(A,b,beta,loss,eps)
    #realization=post(rhoParams)#,num_realizations=20)
    #print(realization.shape)
    eU=empR(realization)
    
    #c=np.mean(eU)
    #print("n_c: ",c).
    #print(type(eU))
    return eU

def empirical_val(A,b,realization,arch,mask,dim,loss,eps,num_realizations=10):
    
    #print(realization.shape)          
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo
    wojtyla= lambda a: pozzo(a,mask,arch)
  
    realization=jax.vmap(wojtyla)(realization)
    empR=lambda beta : return_loss_function(A,b,beta,loss,eps)
    #realization=post(rhoParams)#,num_realizations=20)
    #print(realization.shape)
    eU=empR(realization)
    return eU



#does it make sense to differentiate prior and posterior, when both are gaussian

def get_loss_function(A,b,weights,loss,eps=2.99):


    r_eps=0
    #print(arch)
    #print(A.shape)
    #fun_sq = lambda beta: ffnnLossJ(A,arch,beta,b) 

    fun_map = lambda beta: ffnnLossPozzo(A,beta,b) 

    fun_b = lambda x: (x <= eps).astype(dtype='float32') * x + (x > eps).astype(dtype='float32') * eps
    
    r_eps= fun_b(fun_map(weights))#*m#ABS BOUNDED!
    
    return r_eps
    
 
def return_loss_function(A,b,weights,loss,eps=2.99):

    r_eps=0

    #print(weights.shape)
    #fun_sq = lambda beta: ffnnLoss(A,arch,beta,b) 

    fun_map = lambda beta: ffnnLossPozzo(A,beta,b) 

    fun_b = lambda x: (x <= eps).astype(dtype='float32') * x + (x > eps).astype(dtype='float32') * eps
    
    r_eps= jax.vmap(fun_b)(jax.vmap(fun_map)(weights))#jax.lax(fun_map,weights,batch_size=b.shape[0]/10)))
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


def pacA(piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch): #EXP ON!!

    return lambda beta: pacApprox(beta,piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch)


def pacApprox(params,piParams,eps,delta,a_p_c,a,m,alph,lambda_,p,dim,arch):

    #STOPPED GRADIENT
  
    bb=0
    rhoParams=[params[0],params[1]] #jnp.diag(params[1])]lax.stop_gradient(
    piParams=[piParams[0],piParams[1]] #jnp.diag(piParams[1])]

    NNsize=dimComp(arch)
    kl=KLdiag(piParams,rhoParams,NNsize)
    #print(' kl   ',kl)
    # to be hopefully replaced
    #print('m ',m)
    #print(jnp.log(delta))
    #bb= 2*(jnp.log(delta))/jnp.sqrt(m)# + (.5*eps**2)/jnp.sqrt(m)
    #print('bb',bb)
    #print(A.shape[0],A.shape[1])
    #print('LIP',naiveLip(prior(piParams),arch))
    theta=(naiveLip(prior(piParams),arch)*a_p_c+1)#*alph*jnp.exp(-lambda_*(a-p))  #lambda gamma: 
    #print(theta)
    bb= (1./jnp.sqrt(m))*kl+jnp.sqrt((eps*delta*theta/m)*kl)
    #print('\n\tbound ',bb*1)
    #bound.append(bb[0]*1)
    return bb#jnp.float32(bb)

def pacApproxE(params,piParams,eps,delta,a_p_c,a,m,alph,lambda_,p,dim,arch):

    #STOPPED GRADIENT
    
    p=1
    bb=0
    rhoParams=[params[0],params[1]]#jnp.diag(params[1])]lax.stop_gradient
    piParams=[piParams[0],piParams[1]]#jnp.diag(piParams[1])] 
    NNsize=dimComp(arch)
    kl=KLdiag(piParams,rhoParams,NNsize)
    theta=(naiveLip(prior(piParams),arch)*a_p_c+1)*alph*jnp.exp(-lambda_*(a-p)) 
    #print(jnp.exp(-lambda_*(a-p)))#lambda gamma: 
    bb= (1./jnp.sqrt(m))*kl+jnp.sqrt((eps*delta*theta)*2*kl)
    #print(bb)
    return bb

def pacBound(params,piParams,eps,delta,a_p_c,a,m,alph,lambda_,p,dim,arch):
    p=1
    bb=0
    rhoParams=[params[0],params[1]]#jnp.diag(params[1])]
    piParams=[piParams[0],piParams[1]]#jnp.diag(piParams[1])]
    NNsize=dimComp(arch)
    bb= 2*(jnp.log(delta))/jnp.sqrt(m) + (.5*eps**2)/jnp.sqrt(m)
    kl=KLdiag(piParams,rhoParams,NNsize)
    theta=(naiveLip(prior(piParams),arch)*a_p_c+1)  #lambda gamma: 
    bb= (1./jnp.sqrt(m))*kl+jnp.sqrt((eps*delta*theta/m)*kl)
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
    



def naiveLip(sample,arch):
	
	'''
	Computes the simplest bound for the lipschitz constant of a NN	
	takes as an input the tensor of parameters of the network
	'''
	#params=jnp.reshape(params,size,arch)
	Lip=1
	index=0
	
	for i in range(len(arch)-1):
		#print(index+(arch[i]*(arch[i+1])))
		#print(lax.slice_in_dim(samplePi,index,index+(arch[i]*(arch[i+1])),axis=1).shape,arch[i]*(arch[i+1]))
		#print(params[i,:arch[i],:arch[i+1]].shape)
		Lip = Lip*jnp.linalg.norm(jnp.reshape(lax.slice_in_dim(sample,index,index+(arch[i]*(arch[i+1])),axis=1),(arch[i],arch[i+1])),ord=2)
		index=index+arch[i]*(arch[i+1])+arch[i+1]	
		#print(jnp.linalg.svd(params[i,:arch[i],:arch[i+1]]))#params[i,:arch[i],:arch[i+1]],
	return Lip

def LipC(w,size,arch,N=100):

    #print(w.shape)
    c=0
    for i in range(N):
        c=c+naiveLip(w,arch)
    #print('mcLip\t',c/N)
    return c/N
  
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


def validation(Z,data_val,params,inp,p,c,arch,dim,Ndraws,acrit=.25,rngV=jax.random.key(0)):


  #N=data_val.shape[1]
  x_size=data_val.shape[0]
  val_cone_slicing= lambda beta: lax.dynamic_index_in_dim(data_val, beta,axis=0)
  list_windows= jnp.array([jnp.arange(element-p,element+p+1) for element in range(p,x_size-p-1)])
  
        
  def validator(it_coord,it_params,rngV):
      val_mapped=jax.vmap(val_cone_slicing)(it_coord) 
      val_mapped=jnp.reshape(val_mapped,(len(it_coord),Z.a ))#print('coord',x_coord)
      struct=[inp,*arch]
      mask=[0]
      s=0  
      rngV,subkey=jax.random.split(rngV)
      for el in range(len(struct)-1):
         s+= (struct[el]+1)*struct[el+1]
         mask.append(s)
         #print(mask)   	
      test=data_val[:,-1] 
      coord=[]
      for t in reversed(range(p)): #parallelize!!!
          coord.append(jnp.array([[v, - (t + 1)] for v in range(-c*(t + 1), c*(t + 1) + 1)]))
      coord = jnp.concat(coord, axis=0)
      coord = jnp.expand_dims(jnp.expand_dims(coord, 0), 0)
      
      cone_end_coordinates = jnp.array([p, data_val.shape[1]-1])#, dtype="int32")
      #print("\tbefore ends: ", cone_end_coordinates)
      cone_end_coordinates = jnp.expand_dims(cone_end_coordinates, 0)
      #print("\t\tafter ends: ", cone_end_coordinates)
      cone_coordinates = cone_end_coordinates + coord[0, 0, :, :]
      #print("cone: ", cone_coordinates)
      def gather_nd(par,indices):
          flat_idx=jnp.ravel_multi_index(indices.T,par.shape)
          return par.flatten()[flat_idx]	
      #print(cone_mapped.shape)
      #print('val mapped',val_mapped.shape)
      A=gather_nd(val_mapped,cone_coordinates)
      #print(it_params[0].shape,it_params[1].shape)
      b=gather_nd(val_mapped,cone_end_coordinates)
      w=post([it_params[0],it_params[1]],Ndraws,seed=rngV)    
      #print(w.shape)
      o=ffnnV(A,[inp,*arch],mask,w)
      output=o.reshape((o.shape[1]))
      #print(output)
      #print(b)
      q=lambda x: (b<=x)*0 + (b>x)*1
      s=jax.vmap(q)(output)
      #print(s)
      s=jnp.sum(s)
      return [output,s]          
  vvv=lambda beta,gamma: validator(beta,gamma,rngV)
  output,pit=jax.vmap(vvv)(list_windows,params)
  pit=jnp.bincount(pit.astype('int16'),length=output.shape[1])
  Xtesting= chisquare(f_obs=pit).pvalue
  print('validation at ', acrit,'% level with p-value ',Xtesting)

  return output,pit,Xtesting

#@jit
def OptSGD(Z,x_size,loss,eps,delta,data,inp,p,c,arch,dim,Ndraws,Ncones,Ncoords,shard_size,lr,rhoScaling,slope,q,piScaling=1,acrit=.25):

    struct=[inp,*arch]
  
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
       #print(mask)   	
     
    #max_val=int(data.shape[1]-1)
    last_cone  = data[:,-Z.a:] #change!!!25:50
    data_train = data[:,:-Z.a]#-2001
			
    #validation=data_val[:,0]
    test=data[:,-1]
			
    N=data.shape[1]
    x_size=last_cone.shape[0]
    #print('in sgd',x_size)
    #time.sleep(10)
	#trange(Ndraws, position=0, desc="r", leave=True, colour='green'):rrrrrrr
    
    
    coord = []
    rng = jax.random.key(0)
    #output = np.array(np.zeros((Ncoords,Ndraws)))
    output=[]
    for t in reversed(range(p)): #parallelize!!!
        coord.append(jnp.array([[v, - (t + 1)] for v in range(-c*(t + 1), c*(t + 1) + 1)]))
    coord = jnp.concat(coord, axis=0)
    coord = jnp.expand_dims(jnp.expand_dims(coord, 0), 0)
    
    bounds=[]#np.zeros(Nsteps)
    pars=[]
    #print(coord)  
    list_windows= jnp.array([jnp.arange(element-p,element+p+1) for element in range(p,x_size-p)])
    #print(list_windows)
    aux_in=[inp,*arch[:-1]]
    #print(aux_in)
    piScale=jnp.array([])
    for in_,out_ in zip(aux_in,arch):
      #print(in_,out_)
      layer=jnp.ones((in_+1)*out_)/in_
      piScale=jnp.hstack([piScale,layer])
    #rhoScale=jax.vmap(scaleInit,in_axes=(0,0))(aux_in,jnp.array(arch))
    #print(piScale)
    #rhoParams =[(jnp.ones(dim))*rhoScaling,jnp.ones((dim))*1.5]#]#jnp.array(loc),jnp.array(scale)
    rhoParams=[jnp.ones(dim)/2,jnp.ones(dim)/(Ncones+1)]#
    if piScaling!=1:
      print('fixed setup')
      piScale=jnp.ones(dim)/piScaling
    else: print('mixed setup')
    piParams=[jnp.zeros(dim),piScale] #SET BACK GRID SEARCH INIT!!!
    sharded_params=jax.vmap(lambda dummy: jnp.vstack(rhoParams))(range(shard_size))
    params= [sharded_params for el in range((x_size-2*p-1)//shard_size)]##jnp.tile(jnp.hstack(rhoParams),((x_size-2*p),1)) # correct w/ cone modulation 
    #print('ps',params[0].shape)
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
    opt_state_grid = [jax.vmap(optimizer.init)(sharded_params) for el in range((x_size-2*p-1)//shard_size)]
    #print(opt_state_grid[0])
       
    last_cone_slicing= lambda beta: lax.dynamic_index_in_dim(last_cone, beta,axis=0)
    #for x_coord in range(Ncoords):#,40):#(x_size, position=0,leave=True, desc="coordinate", colour='red'):    
    #@partial(jit,static_argnums=1)
    time_windows=([jnp.arange(jnp.maximum(0,Z.N-Z.a*Z.Ncones*(element+1)),(Z.N-Z.a*Z.Ncones*element)) for element in range(Z.Nbatches)])
    t_slicing= lambda beta: lax.dynamic_index_in_dim(jnp.transpose(data_train), beta,axis=0) 

    def ef(it_coord,data,params,piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch):

        window_mapped=jax.lax.map(d_slicing,(it_coord))#print('coord',x_coord)
        # print('shape',window_mapped.shape,batch)
        window_mapped=jnp.reshape(window_mapped,(len(it_coord),Z.Bsize))
        #print('shape',window_mapped.shape)
        
        cone_mapped= jax.lax.map(last_cone_slicing,(it_coord))
        #print(cone_mapped.shape)
        cone_mapped=jnp.reshape(cone_mapped,(len(it_coord),Z.a ))
        #print(params[0].shape)
        Ac,bc=Z.get_coneJ((window_mapped),sizeData=window_mapped.shape[1])
        scorfval=(r_empirical_risk( Ac,bc,arch,mask,dim,loss,eps,num_realizations=20))  
        key = jax.random.PRNGKey(batch)
        finalSge=scorfval(post([params[0],params[1]],num_realizations=10,seed=key))#jax.jit  	
        sm=jnp.mean(finalSge)   
      
        #print('pac',pacBound(params,piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch))
        
        finalBound= sm + pacBound(params,piParams,eps,delta,a_p_c,a,m,alpha,lambda_,p,dim,arch)    
        return finalBound

    def coordit(it_coord,it_params,opt_state,batch,Bsize,slope,q,rng=rng):
        #if not x_coord%10: print('cood: ',x_coord)
        #x_coord=jnp.where(it_coord<Ncoords-p,it_coord,0)
        #print(it_coord)
        #Z.data=data
        #print(it_params.shape[0])
        #it_params=jnp.reshape(it_params,(2,int(it_params.shape[0]/2)))
        #print('in init coord')   ####################################################
        #print(it_coord)
        window_mapped=jax.vmap(d_slicing)(it_coord)#print('coord',x_coord)
        #print('shape',window_mapped.shape)
        window_mapped=jnp.reshape(window_mapped,(len(it_coord),Bsize))
        #print('shape',window_mapped.shape)
        
        cone_mapped= jax.lax.map(last_cone_slicing,(it_coord))
        #print(cone_mapped.shape)
        cone_mapped=jnp.reshape(cone_mapped,(len(it_coord),Z.a ))
        #print('shape',cone_mapped.shape)
        Acones,bcones=Z.get_coneJ((window_mapped),sizeData=time_windows[batch].shape[0])
        input_size,Z.m=Acones.shape
        num_realizations=1
        #fun=(pacG(Acones,bcones,piParams,eps,delta,Z.a,Z.m,Z.alpha,Z.lambda_,dim,jnp.array([inp,*arch]),loss))  # jax.jit
        
        #Z.data=window_mapped
        #print(Z.data.shape)
        #if x_coord<p or x_coord>x_size-1-p: #if it's outside the cone then skip
        #    return 
        #Z.x_position=tf.constant(x_coord,shape=(1,),dtype=tf.int32)
        cone_end_coordinates = jnp.array([p, last_cone.shape[1]-1])#, dtype="int32")
        #print("\tbefore ends: ", cone_end_coordinates)
        cone_end_coordinates = jnp.expand_dims(cone_end_coordinates, 0)
        #print("\t\tafter ends: ", cone_end_coordinates)
        cone_coordinates = cone_end_coordinates + coord[0, 0, :, :]
        #print("cone: ", cone_coordinates)
        def gather_nd(par,indices):
            flat_idx=jnp.ravel_multi_index(indices.T,par.shape)
            return par.flatten()[flat_idx]	
        #print(cone_mapped.shape)
        A=gather_nd(cone_mapped,cone_coordinates)
        b=gather_nd(cone_mapped,cone_end_coordinates)
        
        funApprox=(pacA(piParams,eps,delta,A.shape[0],Z.a,Z.m,Z.alpha,Z.lambda_,Z.p,dim,([inp,*arch])))  # jax.jit
        sgds=[]#np.zeros((Nsteps,2,len(rhoParams[0])))
        sges=[]#np.zeros(Nsteps)												
        scorf=(l_empirical_risk(Acones[:,:],bcones[:],([inp,*arch]),mask,dim,loss,eps,num_realizations))#jax.jit
        scorfval=(r_empirical_risk(Acones[:,:],bcones[:],([inp,*arch]),mask,dim,loss,eps,num_realizations=1))#jax.jit  	
        value,grads = jax.value_and_grad(funApprox)(it_params) #EXPERIMENTAL VERSION!!! CHANGE
        
        jest = sge_pwj(scorf,it_params,my_multi_normal,rng,num_samples=1)# 10 is just for example, 
        jest = jnp.mean((jest),axis=0)
        updates = jest + grads *0.001
        print('grad',grads,jest)#print(updates)
        #print(opt_state)
        updates, opt_state = optimizer.update(updates,opt_state,it_params) #print(opt_state)
        #print('jest',jest)	
        it_params = optax.apply_updates(it_params,updates)
        #print(it_params)
        sgeVal = scorfval(post([it_params[0],it_params[1]],num_realizations=1,seed=rng)) #10
        sm=jnp.mean(sgeVal)
        cb=value+sm #+sc
        rng,subkey=jax.random.split(rng)
        #time_mapped_rescaled=rescalingInv(time_mapped,slope,q,eps=0.000001)
        
        #print('\nsge\t',batch+1,'\t',sm*1)
        #print('\nec\t',batch+1,'\t',sc*1)
        #print('bound\t\t',cb*1)
        #print(jax.vmap(lambda beta : beta-test)(output))
        
        #A=gather_nd(cone_mapped,cone_coordinates) #rescalingInv(cone_mapped,slope,q,eps=0.000001)rescalingInv(cone_mapped,slope,q,eps=0.000001)
        #b=gather_nd(cone_mapped,cone_end_coordinates)     
        w=post([it_params[0],it_params[1]],Ndraws,seed=rng)    
        #print(w.shape)
        o=ffnnV(A,[inp,*arch],mask,w)
        #print('oshape',o.shape)
        #output.append(o)
        output=o.reshape((o.shape[1]))
        #print(output)
        #print(b)
        q=lambda x: (b<=x)*0 + (b>x)*1
        s=jax.vmap(q)(output)
        #print(s)
        s=jnp.sum(s)
        #print(s)jnp.array
        return [output,it_params,opt_state,s,Acones,bcones]

    #params_shards= jnp.array([jnp.arange(element,(element+shard_size)) for element in range(p,x_size-p-1,shard_size)])
    list_shards= ([jnp.array([jnp.arange(c_coord-p,c_coord+p+1) for c_coord in range(element,element+shard_size)]) for element in range(p,x_size-p-1,shard_size)])
    print(Z.Nbatches)
    for batch in range(1,Z.Nbatches):
    #def batching(batch)#,params=params,opt_state=opt_state,rng=rng):: int,len(range(0,x_size,shard_size))
    #here starts a new batch init
        #subkey,rng=jax.random.split(rng)
        key = jax.random.PRNGKey(batch)
        keys = [jax.random.split(key*(el+1), shard_size).reshape(shard_size, 2) for el in range((x_size-2*p-1)//shard_size)]#print(rng,subkey)
        #print('time',batch)
        time_mapped=jax.vmap(t_slicing)(time_windows[batch])#print('coord',x_coord)
        #print(time_mapped.shape)
        time_mapped=jnp.squeeze(time_mapped,axis=1)      #jnp.reshape(time_mapped,(len(it_coord),Z.Bsize),order='F')
        shard_slicing=lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)
        #Z.data=time_mapped
        #print(time_mapped.shape)
        #print(list_shards)
        d_slicing= lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)  #  [,:]
        ccc=jax.vmap(lambda og,pmap,opt,rngM: coordit(og,pmap,opt,batch,Z.Ncones*Z.a,slope,q,rngM),in_axes=(0,0,0,0))
        #print(time_mapped[:,list_shards[0]]   )
        #window_sharded = jax.vmap(shard_slicing)(list_shards)
        #print(window_sharded.shape)
        pit=jnp.array([])
        output=jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(Ndraws)))
        params_stacked =jnp.transpose(jax.vmap(lambda dummy:jnp.array([[],[]]))(range(dim)))
        #params_stacked = jnp.vstack([params_stacked,params_stacked])
        for ls_i in range((x_size-2*p-1)//shard_size):
          #print(ls_i)#print('ls_i',params[ls_i].shape,ls_i)#print(list_windows)z
          #print(opt_state_grid[ls_i])
          #print(type(opt_state_grid[0].mu))
          #slice_=jnp.arange(ls_i*shard_size,(ls_i+1)*shard_size)
          #opt_i=(opt_state_grid[0].count[slice_],opt_state_grid[0].mu[slice_],opt_state_grid[0].nu[slice_])
          [out,params[ls_i],opt_state_grid[ls_i],pit_vals,Ac,bc]= ccc(list_shards[ls_i],params[ls_i],opt_state_grid[ls_i],keys[ls_i])#jax.jit
          output=jnp.vstack([output,out])
          params_stacked=jnp.vstack([params_stacked,params[ls_i]])
          pit=jnp.hstack([pit,pit_vals]) #stateMap= lambda up, state : optimizer.update(up,state,it_params)
          #updates, opt_states = jax.vmap(stateMap,zip(updates,opt_states))
          #pitdist= jnp.bincount(pit_vals,jnp.ones((x_size,))/x_size).cdf(low=0,high=51)
        #print(params_stacked .shape)
  
        kspit=pit
        pit=jnp.bincount(pit.astype('int16'),length=output.shape[1]+1)
        #print(pit)#25/output.shape[0])
        #,jnp.ones((x_size-2*p,))/(x_size-2*p))
        #if jnp.sum(pit):  #might be modified to speed up the process
        Xtesting= chisquare(f_obs=pit).pvalue
        kstesting = kstest(kspit,randint.rvs(low=0,high=50,size=100)).pvalue  #, f_exp= jnp.ones(pit.shape[0])*50).pvalue
        #print(pit.sum(),Xtesting,kstesting,acrit)
        #else: Xtesting=0#kstest(pit,randint.rvs(low=0,high=50,size=1000) )
        #print('pvalue',kstesting.pvalue)#,'\t pit', pit,'o',output.shape[1])
        #if Xtesting > acrit:# and pit.sum()==x_size-2*p-1 : 
          #break
    print('p-value after ',batch,' iterations: ',Xtesting) 
    finalStep=lambda coord,beta: ef(coord,data,beta,piParams,eps,delta,Ac.shape[0],Z.a,Z.m,Z.alpha,Z.lambda_,Z.p,dim,([inp,*arch]))
    finalBound=jnp.array([])
    for ls_f in range((x_size-2*p-1)//shard_size):   
      fB=jax.vmap(finalStep)(list_shards[ls_f],params[ls_f])
      finalBound=jnp.hstack([finalBound,fB])
    #print(x_size,len(pit_vals))
    #finalPit= jnp.bincount(pit_vals,jnp.ones((x_size-2*p,))/(x_size-2*p))
    print('pit',pit.shape)
    print('stacked params',params_stacked.shape)
    return [output,params_stacked,finalBound,pit,batch,Xtesting]
    
    #return [output,pars,bounds]


    
