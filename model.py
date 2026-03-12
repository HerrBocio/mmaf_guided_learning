import jax.numpy as jnp
from jax import vmap, jit
from utils import dimComp
from functools import partial
import numpy as np
import jax

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
      prior_log_scale =jnp.ones(dim)*jnp.log(config.model.prior_var)/2

      self.pi_params= [prior_mean,prior_log_scale]
    
      init_mean = jnp.ones(dim)*config.model.init_mean 
      init_log_scale= jnp.ones(dim)*jnp.log(config.model.init_var)/2
      self.sharded_params=vmap(lambda dummy: jnp.vstack([init_mean,init_log_scale]))(jnp.arange(self.shard_size))
      self.params= [vmap(lambda dummy: jnp.vstack([init_mean,init_log_scale]))(jnp.arange(config.shard_size)) for el in jnp.arange((config.data.num_coords)//config.shard_size)]   

  
  
  def pozzo(self, weights:jnp.array):
    pozzo=[weights[self.mask[el-1]:self.mask[el]].reshape((self.arch[el-1]+1, self.arch[el])) for el in range(1, len(self.mask))]
    #jax.debug.print("{}")
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
         #jax.debug.print("el {}",el.shape)
         #jax.debug.print("x shape before matmul {}",x.shape)
         x=jnp.matmul(x,el[:-1,:]) +el[-1,:]
         #jax.debug.print("x shape after matmul {}",x.shape)
         activation= lambda vec: (vec<0)*0+(vec>=0)*vec 
         x=vmap(activation)(x)  #activation
         #jax.debug.print("x shape after activation {}",x.shape)
      return jnp.matmul(x,weights[-1][:-1,:])
  
  @partial(jit, static_argnums=0)
  def ffnn_vmap_over_forward_pass(self, inp:jnp.array, weights:list):	
      '''
      ...
      '''
      forward=lambda alpha:(self.ffnn_forward_pass(alpha,weights))
      forward_mapped=vmap(forward,in_axes=1)(inp)
      hX=forward_mapped.reshape((forward_mapped.shape[0]))
      return hX
  
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
      forward=lambda alpha:(self.ffnn_forward_pass(alpha,weights))
      
      forward_mapped=vmap(forward,in_axes=1)(inp)
      forward_mapped=forward_mapped.reshape((forward_mapped.shape[0]))
      l=self.crps_univ_rank_mapped(b, forward_mapped)
      
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

      print('inp',inp.shape)
      parPozzo=lambda alpha: self.pozzo(alpha)
      weights=vmap(parPozzo)(weights)
  
      forward=lambda alpha: self.ffnn_forward_pass(inp,alpha)
      return jnp.transpose(vmap(forward)(weights))


#   @partial(jit, static_argnums=0)
#   def truncated_cov(self, u, tau):
#         """
#         returns Cov(Z_t(x)^(r), Z_{t+tau}(x+u)^(r)) = Var(Lambda') exp(-Au) int_{A_0(0)\V_{(0,0)}^r \cap A_{tau}(u)\V__{(tau,u)}^r} exp(2As) ds
#         """
#          # the formula below works for tau<=0, u in |R. If tau>0, we have to set tau=-tau, u=-u, as Cov(Z_tau(u)^r, Z_0(0)^r) = Cov(Z_0(0)^r, Z_{-tau}(-u)^r) because of stationarity
#         if tau > 0:
#             tau = -tau
#             u = -u
#         #r = a-p
#         if tau <= -self.r:
#             return 0
#         int = self.c_/self.A_ * (-np.exp(-2*self.A_*self.r)*(tau+self.r+1/(2*self.A_)) + np.exp(2*self.A_*tau)/(2*self.A_))

#         return self.VarLevySeed_ * np.exp(-self.A_*u) * int
  
#   @partial(jit, static_argnums=0)
#   def truncated_covs_between_all_members_of_cone(self):
#     """
#     returns: TODO truncated
#         covs_XY: a vector containing the covariances between the apex of the cone Y_i and X_i^{(j)}, j=1,...,a(p,c)
#         covs_XX: a matrix containing Cov(X_i^{(j)},X_i^{(k)}) for all j,k = 1,...,a(p,c)
#     """
#     distances_XY = []
#     for t in reversed(range(self.p)): # t+1 in {p, p-1, p-2, ..., 1}
#         bt = np.floor(self.c_*(t+1)*self.h_t/self.h_s) # b:= argmax {a: a*h_s <= (t+1)*c*h_t}
#         distances_XY.append(jnp.array([[v, -self.h_t*(t+1)] for v in jnp.arange(-bt*self.h_s, (bt+1)*self.h_s, self.h_s)])) # [spatial pos, temporal pos]
#     distances_XY = jnp.concat(distances_XY, axis=0)
#     covs_XY = jnp.array([self.truncated_cov(u=dist[0], tau=dist[1]) for dist in distances_XY])

#     distances_XX = []
#     covs_XX = []
#     for t in range(self.p,0,-1): # t in {p, p-1, p-2, ..., 1}
#         bt = int(np.floor(self.c_*t*self.h_t/self.h_s)) # bt:= argmax {a: a*h_s <= (t+1)*c*h_t}
#         for pixel1 in jnp.arange(-bt*self.h_s,(bt+1)*self.h_s, self.h_s):
#             dist_row = []
#             cov_row = []
#             for s in range(self.p,0,-1):
#                 bs = int(np.floor(self.c_*s*self.h_t/self.h_s))
#                 for pixel2 in jnp.arange(-bs*self.h_s, (bs+1)*self.h_s, self.h_s):
#                     dist_row.append([float(pixel1-pixel2), -self.h_t*(t-s)])
#                     cov_row.append(self.truncated_cov(u = float(pixel1-pixel2), tau = -self.h_t*(t-s)))
#             distances_XX.append(dist_row)
#             covs_XX.append(cov_row)
#     covs_XX = jnp.array(covs_XX)
    
#     return covs_XY, covs_XX

#   @partial(jit, static_argnums=0)
#   def calc_covs(self):
#     self.covs = self.truncated_covs_between_all_members_of_cone()