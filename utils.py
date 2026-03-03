import numpy as np
from jax import random,vmap
import jax.numpy as jnp
from os import path,makedirs

def create_folder(new_path):
  '''
  Creates directory 'new_path'
  '''
  if not path.exists(new_path):
    makedirs(new_path)
    print("folder created")

def get_workspace():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = path.abspath(__file__)
    file = path.dirname(cur_path)
    file = path.dirname(file)
    return file
ws =  get_workspace() + '/mmaf'


def mask_gen(inp,arch):
    '''
    Generates the parameter masking for the nework architecture
    
    '''
    struct=[inp,*arch]
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
    return mask


def makeh5(net,hdata,names):

  '''
  Takes as input the h5 file pointer, data to be stored and the respective names
  Creates and stores data structures in an (already defined!) h5 group
  '''
  for j in range(len(names)):
    net.create_dataset(names[j],data=hdata[j])


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


def get_simulated_data(filename):
    data=loadmat(filename+'.mat')
    data=data["data"]
    return data


def numpy_collate(batch):
  """
  Collate function specifies how to combine a list of data samples into a batch.
  default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
  """
  return jax.tree_util.tree_map(np.asarray, default_collate(batch))


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
    
    sample_shape = (num_realizations,*rhoP[0].shape) 
    sam=random.normal(seed, shape=sample_shape) * jnp.exp(rhoP[1]/2) + rhoP[0]

    return sam



def chi2_diag_gaussians(piParams,rhoParams):
    """
    Function that computes the chi-squared divergence between two gaussian distributions
    Input:
      piParams: parameters of the reference distribution
      rhoParams: parameters of the generalised posterior distribution
    """
    mu_p = rhoParams[0]
    var_p = jnp.exp(rhoParams[1])
    mu_q = piParams[0]
    var_q =jnp.exp(piParams[1])
    delta2 = (mu_p - mu_q)**2
    denom = var_p * (2*var_q - var_p)
    Ii = var_q / jnp.sqrt(denom) * jnp.exp(delta2 / (2*var_q - var_p))
    prod = jnp.prod(Ii)
    return prod 




def KLdiag(piParams,rhoParams,NNsize):
    
    '''
    computes the KL divergence for two multivariate gaussians, with respect to variance
    

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
    kl=jnp.dot(vmap(inv)(piParams1),rhoParams1) 
    kl= kl - NNsize
    diff=piParams0-rhoParams0
    prod=lambda a,b: a*b
    kl= kl + jnp.dot(diff,vmap(prod)(vmap(inv)(piParams1),diff)) 
    kl=kl + jnp.sum(vmap(jnp.log)(piParams1)) 
    kl=kl - jnp.sum(vmap(jnp.log)(rhoParams1)) 
    return kl
    

def KLdiag_from_log_scale(piParams,rhoParams,NNsize):
    
    '''
    computes the KL divergence for two multivariate gaussians, with respect to the log scale
    

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
    #print(rhoParams0.shape,piParams1.shape)
    inv=lambda beta: 1./beta
    kl= jnp.sum(jnp.exp(rhoParams1-piParams1)-1)
    diff=piParams0-rhoParams0
    prod=lambda a,b: a*b
    kl= kl + jnp.dot(diff,vmap(prod)(jnp.exp(-piParams1),diff)) #matmul
    kl=kl + jnp.sum(piParams1) 
    kl=kl - jnp.sum(rhoParams1) 
    return kl/2
    

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
    gradient= vmap(grad)(piParams0,piParams1,rhoParams0,rhoParams1)
    return jnp.transpose(gradient)




def LipC(model,shard_size=int(1e2),N=int(1e3)):

    '''
    Computes the estimation of the Lipchitz constant for a ffnn
    Input:
        piParams: parameters of the reference distribution
        dim: network size
        mask: network layers structure
    Output L
    '''
    parPozzo=lambda alpha: model.pozzo(alpha)
    def naiveLip(weights):
      prod=1
      for el in weights:
        prod*= jnp.linalg.norm(el[:-1,:],ord=2)          
      return prod
    L=0
    for el in range(N//shard_size):
      #print(el.shape)
      realizations = dist_sample(model.pi_params,num_realizations=shard_size,seed=random.key(el))
      w=vmap(parPozzo)(realizations)
      
      c=vmap(naiveLip)(w)
      L+= jnp.sum(c)
      
    L=L/N
    return L
  


def rescalingU1(d,eps=0):
  '''
  Linear rescaling between [-1,1]
  Input
      d: data
      eps: relaxation term
  '''
  m=np.amin(d) 
  M=np.amax(d)
  p=2/(M-m)
  q=(m+M)/(m-M)
  return d*p+q,p,q


def rescalingInv(d,slope,q,eps=0):
  '''
  Inverse linear rescaling to original 
  Input: 
      d: data
      slope: slope relative to initial rescaling
      q: quota relative to initial rescaling
  '''
  return (d - q)/slope

