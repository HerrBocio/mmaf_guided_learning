import pickle
import os
from eval.metrics import Metrics
import h5py
import jax.numpy as jnp
from eval.plots import mmaf_plot

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def load_pickle_file(filepath):
    """
    loads the pickle file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data


def inspect_output_structure(output):
    """
    prints the structure of the loaded dictionary
    """
    
    print("\n=== MODEL ===")
    model=output.get("model")
    print(output.get("model"))
  
    print("\n=== BEST PARAMS ===")
    print(model.best_params)
    
    print("\n=== BEST TRAINING ===")
    best_training = output.get("best_training", {})
    #print("Min iteration:", best_training.get("min_it"))
    print("Min error:", best_training.get("min_error"))
    
    print("\n=== TRAINING HISTORY ===")
    training_history = output.get("training_history", {})
    print("Train error:", training_history.get("train_error"))
    print("PAC:", training_history.get("pac"))
    print("Val JEST:", training_history.get("val_jest"))
    print("Val GRAD:", training_history.get("val_grad"))

    print("\n=== DATA TEST ===")
    print(output.get("data_test"))
    

def ensemble_forecast(filename,a_val,horizon):
    width=[10,10,30,100,300,800]
    depth=[2,5,2,2,2,3]
    priors=[10,30,50,70,90,110,130,150,170,190,210]
    Ndraws=50
    Ncoords=8
    #draws_shard=100
    m=559
    data_path='' #where the original dataset comprised of trend and seasonality is stored
    print(filename)

   

    for i in range(len(width)):
      lowest_crps=+jnp.inf
      lowest_prior=0
      for pi in priors:
        
        filepath = "/output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(int(pi))+'_a'+str(a_val)+".pkl"
    
        output = load_pickle_file(filepath)
        data_x=output['data_test']
        
        model = output["model"]
        metrics=Metrics(model,data_x,Ndraws,filename)
        metrics.multi_ef()
        metrics.crps_univ_rank()
        
        
        if metrics.crps_val_mean<lowest_crps:
          lowest_crps=metrics.crps_val_mean
          lowest_prior=pi

    
      print('Validated prior  [', width[i],'^',depth[i],']:  ',lowest_prior)
      
      filepath = ''+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(int(lowest_prior))+'_a'+str(a_val)+".pkl"
      
      
      output = load_pickle_file(filepath)
      print('min it ',output['best_training']['min_it'])
      validated_model = output["model"]
      target=(output['training_history']['train_error'][output['best_training']['min_it']] + output['training_history']['val_grad'][output['best_training']['min_it']][:,-1]).mean(axis=0)
      validated_metrics=Metrics(validated_model,data_x,Ndraws,filename)
      validated_metrics.multi_ef()
      validated_metrics.crps_univ_rank()
      validated_metrics.rmse_univ_rank()
      
      emp_risk=output['training_history']['train_error'][output['best_training']['min_it']]
      validated_metrics.true_pac(m,emp_risk)

      print('& $['+str(width[i])+'^'+str(depth[i])+']$ &', jnp.round(target,decimals=4),'&',jnp.round(validated_metrics.true_pac_train,decimals=4),'&','$N(0,1/',lowest_prior,')$','&',jnp.round(validated_metrics.crps_val_mean,decimals=4),'&',jnp.round(validated_metrics.mse_val_mean,decimals=4),'&',horizon-1,'&',jnp.round(validated_metrics.crps_test_mean,decimals=4),'&',jnp.round(validated_metrics.mse_test_mean,decimals=4))

      
      mmaf_plot(validated_metrics,lowest_prior,width[i],depth[i],[output['training_history']['val_jest'], output['training_history']['val_grad'],output['training_history']['train_error'] ,output['training_history']['pac']  ],output["epochs"])
