import jax.numpy as jnp
from src.utils import dist_sample
from src.sge_utils import truePAC
from jax import vmap
from jax.random import key
#from STOUNewSetup import ffnnV

class Metrics():

  def __init__(self,model,data,Ndraws,filename):

    data_=jnp.empty((0,data[0].shape[-2],data[0].shape[-1]))
    params_=jnp.empty((0,model.best_params[0].shape[-2],model.best_params[0].shape[-1]))
    for par,el in zip(model.best_params,data):
      data_=jnp.vstack([data_,el])
      params_=jnp.vstack([params_,par])
      #print(data)
    data_=data[0]
    self.filename=filename
    self.data_val = data_[:,:,0].reshape(data_.shape[0],data_.shape[1],1)
    self.data_test= data_[:,:,1:]
    self.data=data[0]
    self.model=model
    self.Ndraws=Ndraws
    self.params=params_
  
  
  def multi_ef_test(self,test_cones,params,Ndraws=1000,Ncones_test=100,rng=60):
    
        w=dist_sample([params[0],params[1]],Ndraws,seed=key(rng))    
        out=lambda A : ffnnV(A,self.model.arch,self.model.mask,w)
        
        m_e_f = vmap(out)(jnp.transpose(test_cones))
        m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        print('mef',m_e_f.shape)
        
        return m_e_f


  def multi_ef_new(self):
    ef=jnp.empty((0,self.Ndraws,19))
    for j in range(self.data_test.shape[0]):
       coord_ef=self.multi_ef_mapped(self.data[j,:-1,:],self.params[j,:,:],rng=10*(j+1))
       ef=jnp.vstack([ef,coord_ef.reshape((1,*coord_ef.shape))])

    self.ef_val=ef[:,:,0].reshape((*ef[:,:,0].shape,1))
    self.ef_test=ef[:,:,1:]
  

  def true_pac(self,m,emp_risk):
    pac=lambda alpha: truePAC(self.model,m,alpha)
    pac_val=vmap(pac, in_axes=0)(self.params)
    self.true_pac_train=(pac_val+emp_risk).mean()

  def multi_ef(self):
    #print('in ef')
    self.ef_val=vmap(self.multi_ef_mapped,in_axes=(0,0,0))(self.data_val[:,:-1,:],self.params,jnp.arange(self.data_val.shape[0]))
    self.ef_test=vmap(self.multi_ef_mapped,in_axes=(0,0,0))(self.data_test[:,:-1,:],self.params,jnp.arange(self.data_test.shape[0]))
   
  def multi_ef_mapped(self,data,params,rng=60):
  
    #print('dataval',data.shape) 
    w=dist_sample(params,self.Ndraws,seed=key(rng))    
    out=vmap(lambda alpha : self.model.ffnnV(alpha,w),in_axes=(1))
    ef= out(data)
    return ef.reshape((ef.shape[-1],ef.shape[0]))
  
  
  def crps_univ_rank(self):

      time_map = lambda alpha,beta: vmap(self.crps_univ_rank_mapped,in_axes=(0,1))(alpha,beta)
      space_map=lambda alpha,beta : vmap(time_map)(alpha,beta) 
      self.crps_val = space_map(self.data_val[:,-1,:],self.ef_val)
      self.crps_val_mean=jnp.mean(self.crps_val)
      self.crps_test= space_map(self.data_test[:,-1,:],self.ef_test)
      self.crps_test_mean=jnp.mean(self.crps_test)

  def crps_univ_rank_mapped(self,y, x):
    
      M=x.shape[0]
      double_sum = vmap(lambda beta: vmap(jnp.abs)(beta-x))
      double_sum = jnp.sum(double_sum(x))
      crps=jnp.sum(jnp.abs(x-y))/M-double_sum/(2*M**2)
      return crps

  def rmse_univ_rank(self):

      time_map = lambda alpha,beta: vmap(self.rmse_univ,in_axes=(0,1))(alpha,beta)
      space_map=lambda alpha,beta : vmap(time_map)(alpha,beta) 
      self.mse_val = space_map(self.data_val[:,-1,:],self.ef_val)
      self.mse_val_mean=jnp.mean(self.mse_val)
      self.rmse_val = space_map(self.data_val[:,-1,:],self.ef_val)
      self.rmse_val_mean=jnp.mean(jnp.sqrt(self.rmse_val))
      self.mse_test= space_map(self.data_test[:,-1,:],self.ef_test)
      self.mse_test_mean=jnp.mean(self.mse_test)
      self.rmse_test= space_map(self.data_test[:,-1,:],self.ef_test)
      self.rmse_test_mean=jnp.mean(jnp.sqrt(self.rmse_test))
  
  def rmse_univ(self,y,x):
      return jnp.mean((x-y)**2)
