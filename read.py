import pickle
import os
from metrics import Metrics
from jax.numpy import inf
from plots import mmaf_plot

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
    

if __name__ == "__main__":

    # Validation
    width=[800]#,300,100,30,10,10]
    depth=[3,2,2,2,2,5]
    priors=[10,30,50,70,90,110,130,150,170,190,210]
    Ndraws=50
    filename="OLR_full"
    
    for i in range(len(width)):
      lowest_mean=inf
      lowest_prior=0
      for pi in priors:
        print('\n\n\t\tset', width[i],pi)
        filepath = "output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(pi)+'_'+str(64)+".pkl"
    
        output = load_pickle_file(filepath)
    
        #inspect_output_structure(output)

        
        model = output["model"]
        metrics=Metrics(model,output['data_test'],Ndraws,filename)
        metrics.multi_ef()
        metrics.crps_univ_rank()
        metrics.rmse_univ_rank()

        if metrics.crps_val_mean<lowest_mean:
          print('aggiornamento',pi,lowest_mean,metrics.crps_val_mean)
          lowest_mean=metrics.crps_val_mean
          lowest_prior=pi

          
    # Test for validated best metrics
      filepath = "output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(lowest_prior)+'_'+str(64)+".pkl"

      output = load_pickle_file(filepath)
      validated_model = output["model"]
      validated_metrics=Metrics(model,output['data_test'],Ndraws,filename)
      validated_metrics.multi_ef()
      validated_metrics.crps_univ_rank()
      mmaf_plot(validated_metrics,lowest_prior,width[i],depth[i])
      
