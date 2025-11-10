import numpy as np
import jax
import jax.numpy as jnp
import os
from jax import random
from scipy.io import loadmat


def create_folder(new_path):
  '''
  Creates directory 'new_path'
  '''
  if not os.path.exists(new_path):
    os.makedirs(new_path)
    print("folder created")


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
    sam=jax.random.normal(seed, shape=sample_shape) * jnp.exp(rhoP[1]/2) + rhoP[0]

    return sam



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
    inv=lambda beta: 1./beta
    kl= jnp.sum(jnp.exp(rhoParams1-piParams1)-1)
    diff=piParams0-rhoParams0
    prod=lambda a,b: a*b
    kl= kl + jnp.dot(diff,jax.vmap(prod)(jnp.exp(-piParams1),diff)) #matmul
    kl=kl + jnp.sum(piParams1) 
    kl=kl - jnp.sum(rhoParams1) 
    return kl
    

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




#def LipC(piParams,dim,mask,arch,shard_size=int(5e2),N=int(1e3)):
def LipC(piParams,dim,mask,arch,shard_size=int(1),N=int(1)):

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
        prod*=jnp.amax(jnp.abs(el))            
      return prod
    L=0
    for el in realizations:
      w=jax.vmap(parPozzo)(el)

      c=jax.vmap(naiveLip)(w)
      L+= jnp.sum(c)
      
    L=L/N
    return L
  


# CRPS
def crps_univ_rank(y, x):

    M=x.shape[0]
    crps=sum(np.linalg.norm(x[j] - y,ord=1) for j in range(M))/M - sum(sum(np.linalg.norm(x[j] - x[k],ord=1) for j in range(M)) for k in range(M))/(2*M**2)
    return crps

def crps_univ_rank_mapped(y, x):

    M=x.shape[0]
    double_sum = jax.vmap(lambda beta: jax.vmap(jnp.abs)(beta-x))
    double_sum = jnp.sum(double_sum(x))
    crps=jnp.sum(jnp.abs(x-y))/M-double_sum/(2*M**2)
    return crps


def rmse_univ_wind(y,x):
    return (jnp.mean(x)-y)**2

def rmse_univ(y,x):
    return jnp.mean((x-y)**2)


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

def rescalingU0(d,eps=0):
  '''
  Linear rescaling between [0,1]
  Input:
      d: data
      eps: relaxation term
  '''
  m=np.amin(d)
  M=np.amax(d)
  p=(1-2*eps)/(M-m)
  q=(M*eps-(1-eps)*m)/(M-m)
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


def multi_ef_validation(test_cones,params,inp,arch,mask,Ndraws,Ncones_test,rng):

        w=dist_sample([params[0],params[1]],Ndraws,seed=rng)    
        out=jax.vmap(lambda A : ffnnV(A,[inp,*arch],mask,w))
        m_e_f = out(test_cones.reshape((Ncones_test,inp)))
        m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        return m_e_f

def multi_ef_test(test_cones,params,inp,arch,mask,Ndraws,Ncones_test,rng):

        w=dist_sample([params[0],params[1]],Ndraws,seed=rng)    
        out=jax.vmap(lambda A : ffnnV(A,[inp,*arch],mask,w))
        m_e_f = out(jnp.transpose(test_cones))
        m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        return m_e_f

  