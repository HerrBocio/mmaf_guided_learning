import pickle
import os
from metrics import Metrics
import h5py
import jax.numpy as jnp
from plots import mmaf_plot

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'

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
    width=[3]#00,100,30,10,10]#,10]
    depth=[2,2,2,2,5,2]
    priors=[10]#,30,50,70,90,110,130,150,170,190,210]
    Ndraws=1000
    filename="Gaudiamonddata1A4mln"
    
    for i in range(len(width)):
      lowest_mean=+jnp.inf
      lowest_prior=0
      for pi in priors:
        
        print('\n\n\t\tset', width[i],pi)
        #filepath = "output/model/Gaudiamonddata1A4mln[800^3]_10_8.pkl"
        filepath = "output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(int(pi))+'_'+str(8)+".pkl"
    
        output = load_pickle_file(filepath)
    
        #inspect_output_structure(output)
        
        model = output["model"]
        print('len',model.best_params[0].shape)
        metrics=Metrics(model,output['data_test'],Ndraws,filename)
        metrics.multi_ef()
        metrics.crps_univ_rank()
        metrics.rmse_univ_rank()
        print(metrics.crps_val.shape,metrics.crps_test.shape)
        
        if metrics.crps_val_mean<lowest_mean:
          lowest_mean=metrics.crps_val_mean
          lowest_prior=pi

          
    # Test for validated best metrics'#
      '''
      file_=h5py.File('/afs/tu-chemnitz.de/project/calibration/siam_results/small/prior10var/01_2801_28_full_relu_std161_Gaudiamonddata1A4mln_a[8, 8]_pir10_m1000_Epoch_60.h5','r')
      m_g=file_.get('m'+str(1000))
      b_g=m_g.get('min')
      
      data_old = jnp.array(m_g.get('data_test'))
      old_params=jnp.array(b_g.get('best params'))

      '''
      print('lowest prior ', width[i],lowest_prior)
      filepath = "output/model/"+filename+'['+str(width[i])+'^'+str(depth[i])+']_'+str(int(lowest_prior))+'_'+str(8)+".pkl"

      output = load_pickle_file(filepath)
      validated_model = output["model"]
      validated_metrics=Metrics(validated_model,output['data_test'],Ndraws,filename)
      validated_metrics.multi_ef()

      mmaf_plot(validated_metrics,lowest_prior,width[i],depth[i])