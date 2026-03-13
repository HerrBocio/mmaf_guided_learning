from jax import grad, jit,vmap,random
import jax.numpy as jnp
from optax import adam,apply_updates
from utils import * 
from sge_utils import *
from tqdm import trange,tqdm
import os
from model import Model
from embedding import Embedding
import pickle


def coordit(model,optimizer,params,batch,opt_state,rng):

  '''
  Core function for the optimization: performs the gradient step and updates the parameters for each spatial coordinate at the current batch iteration
  '''
  

  pac_mapped = lambda alpha: pac_approx(model,alpha,batch[-1].shape[0])
  val_grad=pac_mapped(params)
  #traced=jax.jit(jax.grad(pac_mapped)).trace(params)
  #lowered=traced.lower()
  #compiled=lowered.compile()
  #print('flops pac\n\t', compiled.cost_analysis()[0]['flops'])	#['flops']
  #stochastic gradient estimator for the gradient of the expected empirical risk
  grad= jax.grad(pac_mapped)(params)

  scorf=l_empirical_risk(model,batch)
  val_jest,jest = sge_pwj(scorf,params,my_multi_normal,rng)
  
  updates = jest+grad
  updates, opt_state = optimizer.update(updates,opt_state,params)  
  params = apply_updates(params,updates)
  
  return [params,opt_state,val_jest,val_grad]




def train(config):

      embedding = Embedding(config)
      embedding.embedded_data()
      config.data.a=embedding.a
      
      config.model.inp_size= sum([2*k*embedding.c+1 for k in range(1,embedding.p+1)])
      
      model = Model(config)
      model.Lip=LipC(model)
      
      model.delta=config.hparams.delta
    
      optimizer = adam(config.lr)
      model.opt_state_grid = [jax.vmap(optimizer.init)(model.sharded_params) for el in range(config.data.num_coords//config.shard_size)]
      
      min_it=0 
      min_error=jnp.inf
      milestone_seeds=[]
    
      val_jest_epoch= jnp.empty((0,config.data.num_coords,embedding.Nbatches))
      val_grad_epoch= jnp.empty((0,config.data.num_coords,embedding.Nbatches))
    
      train_epoch= jnp.empty((0,config.data.num_coords))
      pac_epoch  = jnp.empty((0,config.data.num_coords))
      for epoch in trange(1,config.data.epochs+1, desc='epochs', colour='green'):
    
          train_stacked = jnp.array([])
          pac_stacked = jnp.array([])
      
          val_jest_stacked =jnp.empty((0,embedding.Nbatches)) 
          val_grad_stacked =jnp.empty((0,embedding.Nbatches)) 
          for ds_idx,data_shard in enumerate(embedding.clean_data):
              val_jest_batch =  jnp.empty((config.shard_size,0)) 
              val_grad_batch =  jnp.empty((config.shard_size,0)) 
              for batch_idx,batch in enumerate(data_shard):
                  key = jax.random.PRNGKey((batch_idx+1)*(ds_idx+1)*epoch)
                  keys = [jax.random.split(key*(el+1), config.shard_size).reshape(config.shard_size,2) for el in range(config.data.num_coords//config.shard_size)]
                  opt_mapping = jax.vmap(lambda alpha,beta,gamma,phi: coordit(model,optimizer,alpha,beta,gamma,phi),in_axes=(0,0,0,0))
                  
                  [model.params[ds_idx],model.opt_state_grid[ds_idx],val_jest,val_grad]= opt_mapping(model.params[ds_idx],batch, model.opt_state_grid[ds_idx], keys[ds_idx])
    
                  val_jest_batch=jnp.hstack([val_jest_batch,val_jest.reshape(config.shard_size,1)])
                  val_grad_batch=jnp.hstack([val_grad_batch,val_grad.reshape(config.shard_size,1)])
              val_jest_stacked=  jnp.vstack([val_jest_stacked,val_jest_batch])
              val_grad_stacked=  jnp.vstack([val_grad_stacked,val_grad_batch])

              train_shard=jnp.empty((config.shard_size,0))
              for el in range(config.num_realizations//config.shard_realization):
                dist_mapped = lambda alpha,beta: dist_sample(alpha,config.shard_realization,beta)
                
                key = jax.random.PRNGKey((batch_idx+1)*(el+1)*(ds_idx+1)*epoch)
                
                realization = vmap(dist_mapped)(model.params[ds_idx],keys[ds_idx])
                train_shard_realization = lambda alpha,beta: l_empirical_risk(model,alpha)(beta)
                train_shard_realization = vmap(train_shard_realization)(batch,realization)
                train_shard =jnp.hstack([train_shard,train_shard_realization.reshape(config.shard_size,1)])
              
              train_shard=jnp.mean(train_shard,axis=1)
              
              pac_shard   = lambda alpha: pacBound(model,embedding.m_batch,alpha) 
              pac_shard   = vmap(pac_shard)(model.params[ds_idx])
              train_stacked= jnp.hstack([train_stacked,train_shard])
              pac_stacked= jnp.hstack([pac_stacked,pac_shard])
      
          train_epoch=jnp.vstack([train_epoch,train_stacked])
          pac_epoch = jnp.vstack([pac_epoch,pac_stacked])
      
          val_jest_epoch=jnp.vstack([val_jest_epoch,val_jest_stacked.reshape(1,*val_jest_stacked.shape)])
          val_grad_epoch=jnp.vstack([val_grad_epoch,val_grad_stacked.reshape(1,*val_grad_stacked.shape)])
          if jnp.mean(train_stacked+pac_stacked,axis=0)<min_error: 
             min_it=epoch
             min_error=jnp.mean(train_stacked+pac_stacked,axis=0)
             model.best_params=model.params


      output = {
        "data_name": config.data.name,
        "a_val": embedding.a,
        "epochs": config.data.epochs,
        "m": config.data.m_batch,
        "rescaling" : 
        {
            "slope": config.data.slope,
            "quota": config.data.q
        },
        "model": model,
        "best_training" :
        {
          "min_it": min_it,
          "min_error": min_error    
        },
        "training_history":
        {
          "train_error":train_epoch,
          "pac": pac_epoch,
          "val_jest":val_jest_epoch,
          "val_grad":val_grad_epoch
        },
        "data_test":embedding.clean_data_test
      }
        
      with open(config.PATH_MOD+config.data.name+'['+str(config.model.width)+'^'+str(config.model.depth)+']_'+str(config.model.prior_var)+'_'+str(embedding.a)+".pkl", "wb") as f:
        pickle.dump(output, f)




