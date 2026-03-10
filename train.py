from jax import grad, jit,vmap,random
import jax.numpy as jnp
from optax import adam,apply_updates
from utils import * 
from sge_utils import *
from tqdm import trange,tqdm
import os
from easydict import EasyDict as edict
from model import Model
import argparse
from embedding import Embedding
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'#,2,3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'


def get_hparams(eps_default=3, ht_default=1, delta_default=0.025, p_default=1, shard_size_default=1, data_name_default='Gaudiamonddata1A4mln', width_default=10, depth_default=2, init_mean_default=0, init_var_default=.25, prior_mean_default=0, prior_var_default=.1):

  # hyperparameters for mmaf
  parser = argparse.ArgumentParser(description='Entry point of the code')
  parser.add_argument("--eps", type=float, default=eps_default)
  parser.add_argument("--h_t", type=float, default=ht_default)
  parser.add_argument("--delta", type=float, default=delta_default)
  parser.add_argument("-p", type=int, default=p_default)
  
  parser.add_argument("--shard_size", type=int, default=shard_size_default)
  
  parser.add_argument("--data_name", type=str, default=data_name_default)
  
  
  #network shape 
  parser.add_argument("--width", type=int, default=width_default)
  parser.add_argument("--depth", type=int, default=depth_default)

  parser.add_argument("--init_mean", type=float, default=init_mean_default)
  parser.add_argument("--init_sd", type=float, default=init_var_default)
  parser.add_argument("--prior_mean", type=float, default=prior_mean_default)
  parser.add_argument("--prior_sd", type=float, default=prior_var_default)

  args,_ = parser.parse_known_args()
  return args

def default_config(hparams):

  config = edict()
  print(ws)
  config.PATH_MOD = ws + '/output/model/'
  # TODO
  config.PATH_LOG = ws + '/output/log/'
  #config.PATH_FORECAST = ws + '/output/forecast/'

  config.shard_size=hparams['shard_size']
  config.lr=0.005
  config.num_realizations=int(1e3)
  config.shard_realization=int(1e2)
  
  # Data Config
  config.data = edict()
  config.data.name = hparams['data_name']
  config.data.path = ws + '/dataset/'

  
  if config.data.name == 'Gaudiamonddata1A4mln' or config.data.name== 'NIGdiamonddata1A4mln':
      config.data.epochs = 60
      config.data.m_batch=1000
      config.data.num_coords = 8
      config.data.val_start_idx = int(0.999192001*1000001)
      #config.data.test_start_idx=int(0.999192001*1000001)
      config.data.slope = 1
      config.data.q = 0
    
  elif config.data.name =='OLR_full':
      config.data.epochs = 5000
      config.data.m_batch= 36
      config.data.num_coords = 8
      config.data.val_start_idx = int(3520*0.65625) 
      config.data.slope = 0.012402022841533814 
      config.data.q = 0.24020228415338135

  
  config.hparams = edict()
  config.hparams.eps = hparams['eps']
  config.hparams.h_t = hparams['h_t']
  config.hparams.delta = hparams['delta']
  config.hparams.p = hparams['p']

  config.model = edict()
  config.model.width = hparams['width']
  config.model.depth = hparams['depth']

  config.model.init_mean = hparams['init_mean']
  config.model.init_var = hparams['init_sd']

  config.model.prior_mean = hparams['prior_mean']
  config.model.prior_var = hparams['prior_sd']
  
  return config



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




def train():

  width=[10,10,30,100,300,800]
  depth=[2,5,2,2,2,3]
  shard_size=[8,8,8,8,8,2]
  priors=[10,30,50,70,90,110,130,150,170,190,210]
  #[1./10,1./30,1./50,1./70,1./90,1./110,1./130,1./150,1./170,1./190,1./210]
  for i in range(len(width)):
    for pi in priors:
      print('set', width[i],pi)
      hparams = get_hparams(width_default=width[i], depth_default=depth[i], prior_var_default=pi, shard_size_default=shard_size[i])
      hparams = vars(hparams)
      config = default_config(hparams)
      embedding = Embedding(config)
      embedding.embedded_data()
    
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
        
      #with open(config.PATH_MOD+config.data.name+'['+str(config.model.width)+'^'+str(config.model.depth)+']_'+str(config.model.prior_var)+'_'+str(embedding.a)+".pkl", "wb") as f:
      #  pickle.dump(output, f)


#########################    MAIN    ################################

train()




