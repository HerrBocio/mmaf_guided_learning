import jax.numpy as jnp
from jax import vmap, jit
from utils import dimComp
from functools import partial

class Model():

  def __init__(self,config):
          
      arch=[config.model.inp_size,*list([config.model.width for el in range(1,config.model.depth+1)]),1]
      dim=dimComp(arch)

      self.arch= arch
      self.dim=dim
      self.inp_size=config.model.inp_size
      self.shard_size= config.shard_size
      self.eps=config.hparams.eps
    
      self.mask = self.mask_gen()
  
      prior_mean = jnp.ones(dim)*config.model.prior_mean
      prior_log_scale =jnp.ones(dim)*(-jnp.log(config.model.prior_var)/2)

      self.pi_params= [prior_mean,prior_log_scale]
    
      init_mean = jnp.ones(dim)*config.model.init_mean 
      init_log_scale= jnp.ones(dim)*jnp.log(config.model.init_var)/2
      self.sharded_params=vmap(lambda dummy: jnp.vstack([init_mean,init_log_scale]))(range(self.shard_size))
      self.params= [vmap(lambda dummy: jnp.vstack([init_mean,init_log_scale]))(range(config.shard_size)) for el in range((config.data.num_coords)//config.shard_size)]   

  
  
  def pozzo(self, weights:jnp.array):
    pozzo=[weights[self.mask[el-1]:self.mask[el]].reshape((self.arch[el-1]+1, self.arch[el])) for el in range(1, len(self.mask))]
    return pozzo


  def mask_gen(self):
    '''
    Generates the parameter masking for the nework architecture
    '''
    struct=[*self.arch]
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
    return mask
    
  @partial(jit, static_argnums=0)
  def ffnn_forward_pass(self, inp:jnp.array, weights:list):
      '''
      Function that computes the forward pass for a generic neural network
      Input:
          inp: input data for the ffnn
          w: network parameters
      '''
      x=inp
      for el in weights[:-1]:
         x=jnp.matmul(x,el[:-1,:]) +el[-1,:]
         activation= lambda vec: (vec<0)*0+(vec>=0)*vec 
         x=vmap(activation)(x)  #activation
      return jnp.matmul(x,weights[-1][:-1,:])
  
  @partial(jit, static_argnums=0)
  def ffnn_loss_forward_pass(self, inp:jnp.array, weights:list, b:jnp.array, eps:jnp.float32):	
      '''
      Computes montecarlo estimator for the empirical risk of the training data 
      '''
      forward=lambda alpha:(self.ffnn_forward_pass(alpha,weights))
      fun_b = lambda x: (x <= eps).astype(dtype='float32') * x + (x > eps).astype(dtype='float32') * eps
      l=0
      forward_mapped=vmap(forward,in_axes=1)(inp)
      forward_mapped=forward_mapped.reshape((forward_mapped.shape[0]))
      l=vmap(jnp.abs)(forward_mapped-b)
      l=vmap(fun_b)(l)
      l=jnp.mean(l)#-b
      return l
      
      
  @partial(jit, static_argnums=0)
  def crps_loss_forward_pass(self, inp:jnp.array, weights:list, b:jnp.array):
      forward=lambda alpha:(ffnn_forward_pass(alpha,weights))
      
      forward_mapped=vmap(forward,in_axes=1)(inp)
      forward_mapped=forward_mapped.reshape((forward_mapped.shape[0]))
      l=crps_univ_rank_mapped(b, forward_mapped)
      
      return l
  '''
  @partial(jit, static_argnums=0)
  def rmse_loss_forward_pass(self, inp:jnp.array, weights:list, b:jnp.array):
      forward=lambda alpha:(ffnn_forward_pass(alpha,weights))
      
      forward_mapped=vmap(forward,in_axes=1)(inp)
      forward_mapped=forward_mapped.reshape((forward_mapped.shape[0]))
      l=rmse_univ_rank_mapped(b, forward_mapped)
      
      return l
  ''' 
  
  def ffnnV(self, inp:jnp.array, weights:list):

      parPozzo=lambda alpha: self.pozzo(alpha)
      weights=vmap(parPozzo)(weights)
  
      forward=lambda alpha: self.ffnn_forward_pass(inp,alpha)
      return jnp.transpose(vmap(forward)(weights))
