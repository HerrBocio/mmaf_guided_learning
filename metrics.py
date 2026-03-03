import jax.numpy as jnp
from utils import dist_sample
from jax import vmap
from jax.random import key

class Metrics():

  def __init__(self,model,data,Ndraws,filename):

    data_=jnp.empty((0,data[0].shape[-2],data[0].shape[-1]))
    params_=jnp.empty((0,model.best_params[0].shape[-2],model.best_params[0].shape[-1]))
    for par,el in zip(model.best_params,data):
      print(par.shape,el.shape)
      data_=jnp.vstack([data_,el])
      params_=jnp.vstack([params_,par])
    print('data shape',data_.shape)
    self.filename=filename
    self.data_val = data_[:,:,0].reshape(data_.shape[0],data_.shape[1],1)
    self.data_test= data_[:,:,1:]
    self.model=model
    self.Ndraws=Ndraws
    self.params=params_
  
  def multi_ef(self):
    self.ef_val=vmap(self.multi_ef_mapped,in_axes=(0,0,0))(self.data_val[:,:-1,:],self.params,jnp.arange(self.data_val.shape[0]))
    self.ef_test=vmap(self.multi_ef_mapped,in_axes=(0,0,0))(self.data_test[:,:-1,:],self.params,jnp.arange(self.data_test.shape[0]))
    
  def multi_ef_mapped(self,data,params,rng):
  
    print('dataval',data.shape) 
    w=dist_sample(params,self.Ndraws,seed=key(rng))    
    out=vmap(lambda alpha : self.model.ffnnV(alpha,w),in_axes=(1))
    ef= out(data)
    print('ef',ef.shape)
    return ef.reshape((ef.shape[0],ef.shape[-1]))

  
  def crps_univ_rank(self):

      time_map = lambda alpha,beta: vmap(self.crps_univ_rank_mapped,in_axes=(0,0))(alpha,beta)
      space_map=lambda alpha,beta : vmap(time_map)(alpha,beta) 
      self.crps_val = space_map(self.data_val[:,-1,:],self.ef_val)
      self.crps_val_mean=jnp.mean(self.crps_val,axis=0)
      self.crps_test= space_map(self.data_test[:,-1,:],self.ef_test)
      self.crps_test_mean=jnp.mean(self.crps_test,axis=0)

  def crps_univ_rank_mapped(self,y, x):
  
      M=x.shape[0]
      double_sum = vmap(lambda beta: vmap(jnp.abs)(beta-x))
      double_sum = jnp.sum(double_sum(x))
      crps=jnp.sum(jnp.abs(x-y))/M-double_sum/(2*M**2)
      return crps

  def rmse_univ_rank(self):

      time_map = lambda alpha,beta: vmap(self.rmse_univ,in_axes=(0,0))(alpha,beta)
      space_map=lambda alpha,beta : vmap(time_mapped)(alpha,beta) 
      self.rmse_val = space_map(self.data_val[:,-1,:],self.ef_val)
      self.rmse_test= space_map(self.data_test[:,-1,:],self.ef_test)
  
  
  def rmse_univ(self,y,x):
      return jnp.mean((x-y)**2)
  
