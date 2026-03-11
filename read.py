import pickle
import os
from metrics import Metrics
import h5py
import jax.numpy as jnp
from plots import mmaf_plot

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'#,2,3'

def load_pickle_file(filepath):
    """
    loads the pickle file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File non trovato: {filepath}")

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
    

if __name__ == "__main__":

    # Validation
    width=[800]#[10,10,30,100,300]
    depth=[3]#[2,5,2,2,2,3]
    priors=[210,30,50,70,90,110,130,150,170,190,210]
    Ndraws=1000
    draws_shard=100
    filename="OLR_full"
    
    for i in range(len(width)):
      lowest_crps=+jnp.inf
      lowest_prior=0
      for pi in priors:
        
        #filepath = "output/model/Gaudiamonddata1A4mln[800^3]_10_8.pkl"
        filepath = "output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(int(pi))+'_'+str(64)+".pkl"
    
        output = load_pickle_file(filepath)
        data_x=load_pickle_file('OLR_test_cones.pkl')
        #inspect_output_structure(output)

        #print('x',data_x.shape)
        
        model = output["model"]
        metrics=Metrics(model,data_x,Ndraws,filename)
        metrics.multi_ef_new()
        metrics.crps_univ_rank()
        
        
        if metrics.crps_val_mean<lowest_crps:
          lowest_crps=metrics.crps_val_mean
          lowest_prior=pi

    
      print('Validated prior  [', width[i],'^',depth[i],']:  ',lowest_prior)
      filepath = "output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(int(lowest_prior))+'_'+str(64)+".pkl"

      
      
      output = load_pickle_file(filepath)
      validated_model = output["model"]
      target=(output['training_history']['train_error'][output['best_training']['min_it']] + output['training_history']['val_grad'][output['best_training']['min_it']][:,-1]).mean(axis=0)
      validated_metrics=Metrics(validated_model,data_x,Ndraws,filename)
      validated_metrics.multi_ef_new()
      validated_metrics.crps_univ_rank()
      validated_metrics.rmse_univ_rank()
      
      emp_risk=output['training_history']['train_error'][output['best_training']['min_it']]
      validated_metrics.true_pac(36,emp_risk)

      print('& $['+str(width[i])+'^'+str(depth[i])+']$ &', jnp.round(target,decimals=4),'&',jnp.round(validated_metrics.true_pac_train,decimals=4),'&','$N(0,1/',lowest_prior,')$','&',jnp.round(validated_metrics.crps_val_mean,decimals=4),'&',jnp.round(validated_metrics.rmse_val_mean,decimals=4),'& 18 &',jnp.round(validated_metrics.crps_test_mean,decimals=4),'&',jnp.round(validated_metrics.rmse_test_mean,decimals=4))
    
      
      mmaf_plot(validated_metrics,lowest_prior,width[i],depth[i])


