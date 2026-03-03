import pickle
import os
from metrics import Metrics

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
    width=[800,300,100,30,10,10]
    depth=[3,2,2,2,5,2]
    priors=[1./10,1./30,1./50,1./70,1./90,1./110,1./130,1./150,1./170,1./190,1./210]
    Ndraws=50
    filename="Gaudiamonddata1A4mln"
    lowest_mean=+jnp.inf
    lowest_prior=0
    for i in range(len(width)):
      for pi in priors:
        print('\n\n\t\tset', width[i],pi)
        #filepath = "output/model/Gaudiamonddata1A4mln[800^3]_10_8.pkl"
        filepath = "output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(int(1./pi))+'_'+str(8)+".pkl"
    
        output = load_pickle_file(filepath)
    
        #inspect_output_structure(output)

        model = output["model"]
        print('len',model.best_params[0].shape)
        metrics=Metrics(model,output['data_test'],Ndraws)
        metrics.multi_ef()
        metrics.crps_univ_rank()
        mertrics.rmse_univ()
        print(metrics.crps_val.shape,metrics.crps_test.shape)

        if metrics.crps_val_mean<lowest_mean:
          lowest_mean=metrics.crps_val_mean
          lowest_prior=pi

          
    # Test for validated best metrics
      filepath = "output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(int(1./lowest_prior))+'_'+str(8)+".pkl"

      output = load_pickle_file(filepath)
      validated_model = output["model"]
      validated_metrics=Metrics(model,output['data_test'],Ndraws,filename)
      metrics.multi_ef()
      metrics.crps_univ_rank()
      
      
