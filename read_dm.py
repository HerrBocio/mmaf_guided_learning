import pickle
import os
from eval.metrics import Metrics
import h5py
import jax.numpy as jnp
from jax import vmap
from eval.plots import mmaf_plot

from dieboldmariano import dm_test
from tsbootstrap.block_bootstrap import MovingBlockBootstrap
import numpy as np


import matplotlib.pyplot as plt
#from scipy.interpolate import make_interp_spline, BSpline
from matplotlib.ticker import LogLocator, LogFormatter, LogFormatterMathtext


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'


def crps_univ_rank_mapped(y, x):
  
    M=x.shape[0]
    double_sum = vmap(lambda beta: vmap(jnp.abs)(beta-x))
    double_sum = jnp.sum(double_sum(x))
    crps=jnp.sum(jnp.abs(x-y))/M-double_sum/(2*M**2)
    return crps

def mse_univ(y,x):
    return jnp.mean((x-y)**2)


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
    

def ensemble_forecast(tablefile,filename,a_val,horizon,slope,q,lowest_prior,mmaf_size,stg_size,stg_e_size,lstm_size,gru_size):
    Ndraws=50
    m=559
  
    mets=['crps','mse']
    print(filename)
    #### diff ####

    list_coords=range(8)
    Ncoords=8
    filename_diff=day+'_diff'


    x = np.array([10**3, 10**4, 10**5, 10**6])

  
    fig,axs = plt.subplots(2,3,figsize=(15, 8))

    axs[0,0].set_ylabel('CRPS')
    
    axs[1,0].set_ylabel('MSE')

    ######### MMAF ############

    filepath = "/output/model/"+filename+'['+str(mmaf_size[0])+'^'+str(mmaf_size[1])+']_'+str(int(lowest_prior))+'_a'+str(a_val)+".pkl"

    
    output = load_pickle_file(filepath)
    data_x=output['data_test']
    validated_model = output["model"]
    validated_metrics=Metrics(validated_model,data_x,Ndraws,filename,Ncoords=Ncoords)
    validated_metrics.multi_ef()
    validated_metrics.crps_univ_rank()
    validated_metrics.rmse_univ_rank()
    print('mmaf',validated_metrics.mse_test.mean())

    ####### STG ########

  
    file_=h5py.File('/DiffSTG'+filename+'_'+str(stg_size)+'_ef.h5','r')
    m_e_f=np.array(file_.get('ef'))[:,:,0,:,0]
    m_e_f=m_e_f*slope+q
    
    
    stg_metrics=Metrics(validated_model,data_x,m_e_f.shape[-1],filename,Ncoords=Ncoords)

    stg_metrics.ef_val=m_e_f.swapaxes(0,-1)[:Ncoords,:,0].reshape((Ncoords,50,1))
    stg_metrics.ef_test=m_e_f.swapaxes(0,-1)[:Ncoords,:,:]
    stg_metrics.crps_univ_rank()
    stg_metrics.rmse_univ_rank()
    print('stg',stg_metrics.mse_test.mean())
  
    file_.close()

    ###### EMB #######
    print('\t\tDiff embedded')
    file_=h5py.File('/DiffSTG'+filename+'_embedded_'+str(stg_e_size)+'_ef.h5','r')
    #print(file_.keys())
    m_e_f_emb=np.array(file_.get('ef'))[:,:,0,:,0]
    #print(m_e_f)
    m_e_f_emb=m_e_f_emb*slope+q
    print(m_e_f_emb.shape)

    
    stg_metrics_emb=Metrics(validated_model,data_x,m_e_f.shape[-1],filename,Ncoords=Ncoords)
    stg_metrics_emb.ef_val=m_e_f_emb.swapaxes(0,-1)[:Ncoords,:,0].reshape((Ncoords,50,1))

    stg_metrics_emb.ef_test=m_e_f_emb.swapaxes(0,-1)[:Ncoords,:,:]
    #print(stg_metrics.ef_test.shape,stg_metrics.ef_val.shape)
    stg_metrics_emb.crps_univ_rank()
    stg_metrics_emb.rmse_univ_rank()
    file_.close()
    ######## LSTM #########
    print('\t\tLSTM')
    file_=h5py.File('/LSTM/H5/'+filename+str(lstm_size)+'.h5','r')
    #print(file_.keys())
    m_e_f_lstm=np.array(file_.get('ef'))[:,:,:Ncoords]
    m_e_f_lstm=m_e_f_lstm*slope+q

    
    lstm_metrics=Metrics(validated_model,data_x,m_e_f_lstm.shape[-1],filename,Ncoords=Ncoords)

    lstm_metrics.ef_val=m_e_f_lstm.swapaxes(0,-1)[:Ncoords,:,0].reshape((Ncoords,m_e_f_lstm.shape[1],1))
    print(lstm_metrics.ef_val.shape)
    lstm_metrics.ef_test=m_e_f_lstm.swapaxes(0,-1)[:Ncoords,:,:]
    lstm_metrics.crps_univ_rank()
    lstm_metrics.rmse_univ_rank()
    print('lstm',lstm_metrics.mse_test.mean())
    
    file_.close()
    ######## GRU #########
    print('\t\tGRU')
    file_=h5py.File('/GRU/H5/'+filename+str(gru_size)+'.h5','r')
    m_e_f_gru=np.array(file_.get('ef'))#[:,:,0,:,0]
    m_e_f_gru=m_e_f_gru*slope+q

    
    gru_metrics=Metrics(validated_model,data_x,m_e_f_gru.shape[-1],filename,Ncoords=Ncoords)

    gru_metrics.ef_val=m_e_f_gru.swapaxes(0,-1)[:Ncoords,:,0].reshape((Ncoords,m_e_f_gru.shape[1],1))
    gru_metrics.ef_test=m_e_f_gru.swapaxes(0,-1)[:Ncoords,:,:]
    gru_metrics.crps_univ_rank()
    gru_metrics.rmse_univ_rank()
    print('gru',gru_metrics.mse_test.mean())
    
    ########### MODEL COMPARISONS ############
    # mmaf vs. diff -> mmaf vs. dif emb. -> diff vs. diff emb. 

    comp_list = [validated_metrics,lstm_metrics,gru_metrics,stg_metrics,stg_metrics_emb]
    
    comp_list_name = ['mmaf','lstm','gru','stg','stg emb']
    
    tablefile.write('#####################################\n\n'+filename+'\n\n')
    
    comparisons=[]
    for met_id,met in enumerate(mets):
      print(met)
      for idx_1 in range(1):
        comparisons_met=[]
        
        for idx_2 in range(5):
      
          if idx_1 == idx_2: continue
          print('\n',met,'\n',comp_list_name[ idx_1],' vs. ',comp_list_name[ idx_2])
          
          if met=='crps':
            
            ########### CRPS ################
            loss=crps_univ_rank_mapped
            s_met=comp_list[idx_1].crps_test.mean(axis=0) - comp_list[idx_2].crps_test.mean(axis=0)
            #s_crps=np.sign(validated_metrics.crps_test.mean(axis=0) - stg_metrics.crps_test.mean(axis=0) )
          elif met=='mse':
    
            ##########  MSE #################
            loss = mse_univ
            s_met = comp_list[idx_1].mse_test.mean(axis=0) - comp_list[idx_2].mse_test.mean(axis=0)
    
          else:
            raise NotImplementedError('undefined metric')
          
          print( len(s_met[s_met<0]) )
          #print(s_met)
    
          
          # Instantiate the bootstrap object
          n_bootstraps =500
          block_length = 15
          rng = 42
           
          # Generate bootstrapped samples
          return_indices = False
          
          diff_mbb = MovingBlockBootstrap(
            n_bootstraps=n_bootstraps, rng=rng, block_length=block_length
          )
          diff_bs = diff_mbb.bootstrap(s_met,return_indices=return_indices)
          diff_bootstrapped = []
          
          for data in diff_bs:
              diff_bootstrapped.append(data)
          diff_bootstrapped = np.array(diff_bootstrapped)
          
          q_025=np.round(np.quantile(diff_bootstrapped.mean(axis=1),q=0.025,axis=0),decimals=4)
          q_975=np.round(np.quantile(diff_bootstrapped.mean(axis=1),q=0.975,axis=0),decimals=4)
          print('$[',q_025[0],' , ',q_975[0],']$ &')
          print('$',s_met.mean(),'$ &')
    
          
    
          ########## DM TEST ###########

          test=dm_test(comp_list[idx_1].data_test[:,-1,:].swapaxes(0,-1),comp_list[idx_1].ef_test.swapaxes(0,-1).swapaxes(-1,1),comp_list[idx_2].ef_test.swapaxes(0,-1).swapaxes(-1,1), loss= loss,one_sided=True)
          print('\n',test[1])
          comparisons_met.append( ['& $[',q_025[0],q_975[0],']$ & ',np.round(float(s_met.mean()),decimals=4),'& ',np.round(float(test[1]),decimals=6)])
        printline=[item for sublist in comparisons_met for item in sublist]
        
        print('\n',met,'\n',comp_list_name[ idx_1],file=tablefile)
        
        print([str(el) for el in printline],file=tablefile)


      comparisons.append(comparisons_met)
      
      
tablefile= open('comptab.txt','w')
    


ensemble_forecast(tablefile,'Gaudiamonddata1A4mln',8,100,slope=1,q=0,lowest_prior=90,mmaf_size=[100,2],stg_size=1,lstm_size=150,gru_size=2)
ensemble_forecast(tablefile,'NIGdiamonddata1A4mln',8,100,slope=1,q=0,lowest_prior=210,mmaf_size=[30,2],stg_size=1,lstm_size=150,gru_size=10)

ensemble_forecast(tablefile,'2mT',146,40,slope=1,q=0,lowest_prior=10,mmaf_size=[30,2],stg_size=1,stg_e_size=11,lstm_size=2,gru_size=18)

tablefile.close()
