import numpy as np
from jax import random,vmap
import jax.numpy as jnp
from os import path,makedirs
import jax

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
    

def KLdiag_from_log_scale(piParams,rhoParams):
    
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


def naiveLip(weights):
    prod=1
    for el in weights:
        prod*= jnp.linalg.norm(el[:-1,:],ord=2)          
    return prod

def Lip_realizations_masked(realizations_masked):

    '''
    takes masked realizations, outputs the Lipschitz constant of each of them
    '''
    Lips=jax.vmap(naiveLip)(realizations_masked)
    return Lips

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

def truncated_cov(A, c, r, u, tau, VarLevySeed):
    """
    returns Cov(Z_t(x)^(r), Z_{t+tau}(x+u)^(r)) = Var(Lambda') exp(-Au) int_{A_0(0)\V_{(0,0)}^r \cap A_{tau}(u)\V__{(tau,u)}^r} exp(2As) ds

    for u=tau=0, this is Var(Z_t(x)^(r))
    """
        # the formula below works for tau<=0, u in |R. If tau>0, we have to set tau=-tau, u=-u, as Cov(Z_tau(u)^r, Z_0(0)^r) = Cov(Z_0(0)^r, Z_{-tau}(-u)^r) because of stationarity
    if tau > 0:
        tau = -tau
        u = -u
    #r = a-p
    if tau <= -r:
        return 0
    int = c/A * (-np.exp(-2*A*r)*(tau+r+1/(2*A)) + np.exp(2*A*tau)/(2*A))

    return VarLevySeed * np.exp(-A*u) * int

def truncated_covs_between_all_members_of_cone(A, c, h_s, h_t, r, p, VarLevySeed):
    """
    returns: 
        covs_XY: a vector containing the covariances between the apex of the cone Y_i and X_i^{(j)}, j=1,...,a(p,c)
        covs_XX: a matrix containing Cov(X_i^{(j)},X_i^{(k)}) for all j,k = 1,...,a(p,c)
        for the truncated MMAF Z_t(x)^(r)
    """
    distances_XY = []
    for t in reversed(range(p)): # t+1 in {p, p-1, p-2, ..., 1}
        bt = np.floor(c*(t+1)*h_t/h_s) # b:= argmax {a: a*h_s <= (t+1)*c*h_t}
        distances_XY.append(jnp.array([[v, -h_t*(t+1)] for v in jnp.arange(-bt*h_s, (bt+1)*h_s, h_s)])) # [spatial pos, temporal pos]
    distances_XY = jnp.concat(distances_XY, axis=0)
    covs_XY = jnp.array([truncated_cov(A=A, c=c, r=r, VarLevySeed= VarLevySeed,u=dist[0], tau=dist[1]) for dist in distances_XY])

    distances_XX = []
    covs_XX = []
    for t in range(p,0,-1): # t in {p, p-1, p-2, ..., 1}
        bt = int(np.floor(c*t*h_t/h_s)) # bt:= argmax {a: a*h_s <= (t+1)*c*h_t}
        for pixel1 in jnp.arange(-bt*h_s,(bt+1)*h_s, h_s):
            dist_row = []
            cov_row = []
            for s in range(p,0,-1):
                bs = int(np.floor(c*s*h_t/h_s))
                for pixel2 in jnp.arange(-bs*h_s, (bs+1)*h_s, h_s):
                    dist_row.append([float(pixel1-pixel2), -h_t*(t-s)])
                    cov_row.append(truncated_cov(A=A, c=c, r=r, VarLevySeed= VarLevySeed,u = float(pixel1-pixel2), tau = -h_t*(t-s)))
            distances_XX.append(dist_row)
            covs_XX.append(cov_row)
    covs_XX = jnp.array(covs_XX)

    return covs_XY, covs_XX

def theta_r(A,c, r, VarLevySeed):
    return jnp.sqrt(VarLevySeed*(c*r/A + c/(2*A**2)))*jnp.exp(-A*r)
