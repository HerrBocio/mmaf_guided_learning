import os
import platform

import jax
import jax.numpy as jnp

# -------------------------------------------------
# Device detection
# -------------------------------------------------


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")
os.environ.setdefault("JAX_PLATFORM_NAME", "")  # allow auto detect




DEVICES = jax.devices()
DEVICE_COUNT = len(DEVICES)
DEFAULT_BACKEND = jax.default_backend()

print("JAX version:", jax.__version__)
print("Platform:", platform.platform())
print("Backend:", DEFAULT_BACKEND)
print("Device count:", DEVICE_COUNT)

for i, d in enumerate(DEVICES):
    print(f"Device {i}: {d}")

# -------------------------------------------------
# Device helpers
# -------------------------------------------------

def get_devices():
    """Return all available accelerator devices"""
    return jax.devices()

def get_primary_device():
    """Return default device"""
    return jax.devices()[0]

def replicate(x):
    """Replicate data across devices"""
    return jax.device_put_replicated(x, jax.devices())

def shard(x):
    """Shard array across devices"""
    return jax.device_put_sharded(list(x), jax.devices())

# -------------------------------------------------
# Multi-device helpers
# -------------------------------------------------

PMAP_AVAILABLE = DEVICE_COUNT > 1

if PMAP_AVAILABLE:
    print("Multi-device execution enabled (pmap)")
else:
    print("Single device execution")

#--------------------------------------------------
# Argument parsing
#--------------------------------------------------


from easydict import EasyDict as edict
from model import Model
import argparse
from embedding import Embedding
import pickle
from src.utils import ws,create_folder
from eval.read import ensemble_forecast
from train_unbounded import train_unbounded

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
  #TODO
  config.PATH_LOG = ws + '/output/log/'
  create_folder(config.PATH_MOD)
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
      config.horizon=101
      config.data.slope = 1
      config.data.q = 0
    
  elif config.data.name =='OLR_full':
      config.data.epochs = 5000
      config.data.m_batch= 36
      config.data.num_coords = 8
      config.data.val_start_idx = int(2304) 
      config.horizon=19
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




if __name__ == '__main__':

  
  width=[4]#10,10,30,100,300,800]
  depth=[2,5,2,2,2,3]
  shard_size=[8,8,8,8,8,2]
  priors=[10]#,30,50,70,90,110,130,150,170,190,210]
  for i in range(len(width)):
    for pi in priors:
      print('set', width[i],pi)
      hparams = get_hparams(width_default=width[i], depth_default=depth[i], prior_var_default=pi, shard_size_default=shard_size[i])
      hparams = vars(hparams)
      config = default_config(hparams)
      train_unbounded(config)
  ensemble_forecast(config.data.name,config.data.a,config.horizon)
