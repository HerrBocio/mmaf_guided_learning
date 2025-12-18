from params import *
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
#from STOUpozo import STOU
#data_path=""
normalize_data=False
use_different_eps=False
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'
from scipy.stats import kstest as kstest
from scipy.stats import randint 
from tqdm import trange
#from multivariateRank import get_prerank
from SGE import error,truePAC,LipC
from STOU import ffnn_forward_pass,ffnn_loss_forward_pass
import jax
import jax.numpy as jnp
from utils import *
data_path="datasets_2D/" # local
#data_path="../datasets_2D/" # server

#/LOCAL/jasst/results/nopreT/"#'/afs/tu-chemnitz.de/project/calibration/jasminDebug/'

#
#path='/afs/tu-chemnitz.de/project/calibration/aistatsResults/s_data_old/'
#path='/afs/tu-chemnitz.de/project/calibration/s_data_new/'



pretraining=False
if pretraining:
  Epochs=range(1,151) #
  Nepochs=150#len(Epochs)
  preTlabel='_preT'
else:
  Epochs=range(1,epochs_nopreT+1) # aka 7 #range(1,61) #
  Nepochs=epochs_nopreT #60#len(Epochs)
  preTlabel=''
  

file_name= day+'_full_relu_std'+preTlabel#full_tanh_new_setup,full_relu_new_setup
figure_name="depth_relu_std"  #day+'_depth_relu_std'#+preTlabel


a_val=[8,8]
#[p]

delta=0.025
A_estimatedM=[3.840956,3.868912]#,]#,[3.840956]#
c_estimatedM=[1,1]



piRescaling=list(range(10,230,20))


m_batches = [m_batches]
Nbatches=124 # for N=1e6 and m=1e3
Ncoords=10
Ncones_test=101

Ncrps=100


calibration=False
start=10-1


#constants_train=2*np.log(1./0.025)/np.sqrt(m_batches[0]) + eps**2/(2*np.sqrt(m_batches[0]))

#constants_test=2*np.log(1./0.025)/np.sqrt(Ncones_test-1) + eps**2/(2*np.sqrt(Ncones_test-1))

#constants_test_t=np.log(1./0.025)/np.sqrt(Ncones_test-1) + eps**2/(2*np.sqrt(Ncones_test-1))

pre_ranks=["multivariate_rank","average_rank","band_depth","mean","variance","energy_score"]
arch_x_val=[1e4,1e5] #1,1e1,1e2,
arch_legend=['epoch1','epoch10','epoch20','epoch30','epoch40','epoch50']




for current_id in reversed(range(len(datasetsM))):
    
    data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
    data_validation=data[:Ncoords,-(Ncones_test)*a_val[current_id]:-(Ncones_test-1)*a_val[current_id]]
    #print(data_validation)
    b_validation=data_validation[:,-1]
    
    alph=.5/A_estimatedM[current_id]

    
    for i,arch in enumerate((archs)):#reversed 
            fVal=open(pathT+'Table_validation_'+figure_name+'_'+datasetsM[current_id]+str(dimComp([inp,*archs[i]]))+'_'+'.txt','w')
            fVal.write('Train Err. & CRPS & RMSE\n\n')

            fTrain=open(pathT+'Table_training_'+figure_name+'_'+datasetsM[current_id]+str(dimComp([inp,*archs[i]]))+'_'+'.txt','w')
            fTrain.write('Training E. & Lin. PAC & True PAC\n\n')
                            
            
            
            fTest=open(pathT+'Table_TestTrain_'+figure_name+'_'+datasetsM[current_id]+str(dimComp([inp,*archs[i]]))+'_'+'.txt','w')
            fTest.write('KL train & Chi2 & pac lin & true pac  &  CRPS & RMSE & min it  \n\n') #Test Err. &
        

            mask=mask_gen(inp,arch)
            d=dimComp([inp,*archs[i]])
            print(d)
            for pir in piRescaling:
                  print('pir',pir)
                  log_pir=-jnp.log(pir)
                  print(log_pir)
                  piScale=jnp.ones(d)*log_pir
                  if pretraining:
                    
                    init_scale=0.0016
                    init_log_scale=jnp.log(init_scale)
    
                    initParams = [jnp.zeros(d),jnp.ones(d)*init_log_scale] 
                  
                    
                    piMean= dist_sample(initParams)
                    piParams=[piMean[0],piScale]
                  else:
                    piParams=[jnp.zeros(d),piScale]
                  #print(piParams)

                  #Lip=LipC(piParams,d,mask,[inp,*arch])

              
                  print('pir', pir,'\n',file=fVal)
                  print('pir', pir,'\n',file=fTest)
              
                  print(pir,arch,datasetsM[current_id])
                  pathPrior = os.path.join(
                      pathF,
                      datasetsM[current_id],
                      'prior' + str(pir),
                      day,
                      str(dimComp([inp, *arch]))
                  )
                  if not os.path.exists(pathPrior):
                      os.makedirs(pathPrior)
                      print("figures folder created")

                  empirical_obj=np.array([])
                  pac_obj=np.array([])
              
                  bound_train_E=[]
                  bound_train_sup=[]
                  train_errors=[]

                  bound_test_E=[]
                  bound_test_sup=[]
                  test_errors=[]
                  
                  rank_mvr_stacked=np.array([])
                  rank_avr_stacked=np.array([])
                  rank_bd_stacked=np.array([])
                  rank_mean_stacked=np.array([])
                  rank_var_stacked=np.array([])
                  rank_es_stacked=np.array([])

                  rank_mvr_epoch=np.array([])
                  rank_avr_epoch=np.array([])
                  rank_bd_epoch=np.array([])
                  rank_mean_epoch=np.array([])
                  rank_var_epoch=np.array([])
                  rank_es_epoch=np.array([])
                  
                  for m in range(len(m_batches)): 
                    filename = os.path.join(
                        path,
                        'prior' + str(pir) + 'var',
                        file_name + str(dimComp([inp, *arch])) + '_' +
                        str(datasetsM[current_id]) + '_a' + str(a_val) +
                        '_pir' + str(pir) + '_m' + str(m_batches[m]) +
                        '_Epoch_' + str(Nepochs) + '.h5'
                    )
                    try:
                      hf = h5py.File(filename, 'r')
                    except OSError as e:
                       print(filename," FEHLER")
                       continue
                    #hf=h5py.File(path+'prior' + str(pir) +'var/' +file_name +str(dimComp([inp,*arch])) +'_' +str(datasetsM[current_id]) +'_a' +str(a_val) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.h5','r')
                    m_g=hf.get('m'+str(m_batches[m]))                    
                    print(hf.keys())
                    print(m_g.keys())
                    last_params=np.array(m_g.get('last params'))
                    b_g=m_g.get('min')
                    #print(b_g.keys())
                    best_params=jnp.array(b_g.get('best params'))
                    #print(last_params.shape)
                    min_error=np.array(b_g.get('min_error'))#-constants_train
                    min_it=np.array(b_g.get('min iteration'))
                    data_test=np.array(m_g.get('data_test'))
                    #print(data_test.shape)
                    e_g=m_g.get('epoch'+str(min_it))
                    bound_min_E=np.array(e_g.get('bound_test_E'))#-constants_test)
                    bound_min_sup=np.array(e_g.get('bound_test_sup'))#-constants_test)
                    emp_min=np.array(e_g.get('test_errors'))
                    test_min_E=jnp.mean(bound_min_E+emp_min)
                    test_min_sup=jnp.mean(bound_min_sup+emp_min)
                    pac_test_min= test_min_E# + constants_test_t
                    b_test=np.array(e_g.get('cones_test'))

                    
                    rng=jax.random.key(Nepochs)
                    
                    e_g=m_g.get('epoch'+str(min_it))
                  
                    emp_risk =np.array( e_g.get('val_jest'))
                    emp_risk=np.mean(emp_risk)
                    kl_mapped=lambda beta: KLdiag_from_log_scale(piParams,beta,d)#(piParams,rhoParams,NNsize):

                    KL=jax.vmap(kl_mapped)(best_params)
                    KL=jnp.mean(KL)
                    print('kl ',KL)
                    #chi2_gau_map=lambda beta: chi2_diag_gaussians(beta,piParams)
                    #ChiSq=jax.vmap(chi2_gau_map)(best_params)
                    #Chi2=jnp.mean(ChiSq)
                    #print('chi2 ',Chi2)
                    #truePacFun=jax.vmap(lambda beta,gamma: truePAC(beta,piParams,eps,Lip,delta,inp,Ncones_test,alph,p,d,arch,gamma),in_axes=(0,0))
                    #true_bound=truePacFun(best_params,ChiSq)
                    true_bound_E = bound_min_E #?? TODO
                    true_bound_sup = bound_min_sup #?? TODO
                    safe_mean=lambda beta: beta/best_params.shape[0]
                    #chi_mean=jax.vmap(safe_mean)(ChiSq)
                    #Chi2=jnp.sum(chi_mean)
                    #print('chi2',Chi2)
                    print("emp_min + true_bound_E:", emp_min + true_bound_E)
                    print("emp_min + true_bound_sup:", emp_min + true_bound_sup)
                    pac_true_min_E =jnp.mean( true_bound_E+emp_min)
                    pac_true_min_sup =jnp.mean( true_bound_sup+emp_min)
                    print('true pac E',pac_true_min_E)   
                    print('true pac sup',pac_true_min_sup)   
                    ##### TRAIN #####
                    
                     
                    m_e_f_validation=np.zeros((0,Ndraws,1))
                    #print(data_validation.shape)
                    for n1 in range(p*c,data_validation.shape[0]-p*c):
                       coord_e_f=multi_ef_validation(data_validation[n1-p*c:n1+p*c+1,-2],best_params[n1-p*c,:,:],inp,arch,mask,Ndraws,1,rng)
                       #print(i,coord_e_f.shape)
                       m_e_f_validation=np.vstack([m_e_f_validation,coord_e_f.reshape((1,*coord_e_f.shape))])
                    #print(np.amax(m_e_f[0,:,:],axis=0).shape)
                    crps=[]
                    rmse=[]
                    for n1 in range(8):
                      crps_coord=[]
                      rmse_coord=[]
                      #for j in range(b_test.shape[-1]):
                      crps_coord.append(crps_univ_rank_mapped(b_validation[n1],m_e_f_validation[n1,:,0]))
                      crps.append(crps_coord)
                      rmse_coord.append(rmse_univ(b_validation[n1],m_e_f_validation[n1,:,0]))
                      rmse.append(rmse_coord)
                      
                    #print(np.array(crps[0]).shape)
                    qs=[.025,.05,.1,.15,.2,.25,.3,.35,.4,.45]#np.linspace(0.05,.45,10)##np.linspace(0.05,.45,10)
                    qs_label=['5%','10%','20%','30%','40%','50%','60%','70%','80%','90%']#print(crps)
       
                    print(np.round(min_error,decimals=4),'&',np.round(np.mean(crps),decimals=4),'&',np.round(np.mean(np.sqrt(rmse)),decimals=4),'&',min_it,file=fVal)    
                    print(np.round(min_error,decimals=4),'&','&',min_it,file=fTrain)    
                    


                    ###### TEST #######

                    m_e_f_test=np.zeros((0,Ndraws,Ncones_test-1))
                    #print(data_test.shape)
                    for i in range(p*c,data_test.shape[0]-p*c):
                       coord_e_f=multi_ef_test(data_test[i-p*c:i+p*c+1,-(Ncones_test-1)*a_val[current_id]-2::a_val[current_id]],best_params[i-p*c,:,:],inp,arch,mask,Ndraws,Ncones_test-1,rng)
                       #print(i,coord_e_f.shape)
                       m_e_f_test=np.vstack([m_e_f_test,coord_e_f.reshape((1,*coord_e_f.shape))])
                    #print(m_e_f_test )
                    #print(np.amax(m_e_f[0,:,:],axis=0).shape)
                    crps=[]
                    rmse=[]
                    for n1 in range(8):
                      crps_coord=[]
                      rmse_coord=[]
                      
                      for n2 in range(b_test[:,:Ncrps].shape[-1]):
                        crps_coord.append(crps_univ_rank_mapped(b_test[n1,n2],m_e_f_test[n1,:,n2]))
                        crps.append(crps_coord)
                        rmse_coord.append(rmse_univ(b_test[n1,n2],m_e_f_test[n1,:,n2]))
                        rmse.append(rmse_coord)
                      
                    #print(np.array(crps[0]).shape)
                    qs=[.025,.05,.1,.15,.2,.25,.3,.35,.4,.45]#np.linspace(0.05,.45,10)##np.linspace(0.05,.45,10)
                    qs_label=['5%','10%','20%','30%','40%','50%','60%','70%','80%','90%']#print(crps)
                      
                    
                    #print(np.round(KL,decimals=4),'&',Chi2,'&',np.round(pac_test_min,decimals=4),'&' , np.round(pac_true_min,decimals=4),'&',np.round(np.mean(crps),decimals=4),'&',np.round(np.sqrt(np.mean(rmse)),decimals=4),'&',min_it, file=fTest)    #'&',np.round(emp_risk,decimals=4)

  

                    
                    for Epoch in Epochs:
                        #print(Epoch)
                        #print(m_g.keys())
                        
                        e_g=m_g.get('epoch'+str(Epoch))
                        
                        #print(e_g.keys())
                        #e_f= np.array(e_g.get('e_f'))
                        #data_test=np.array(e_g.get('cones_test'))
                        #print(data_test[:,0])
                        #e_f_inv= np.array(e_g.get('e_f_inv'))
                        bound_train_E.append(np.mean( np.array(e_g.get('bound_train_E'))))#-constants_train)
                        bound_train_sup.append(np.mean( np.array(e_g.get('bound_train_sup'))))#-constants_train)
                        train_errors.append( np.mean(np.array(e_g.get('train_errors'))))
                        #min_bound_train.append(np.mean( np.array(e_g.get('min bound'))))
                        
                        #params=np.array(e_g.get('params'))
                        #arams=jnp.
                        #print(params[0].shape)
                        #print(np.array(e_g.get('best_paramsmin error')))
                        bound_test_E.append(np.mean( np.array(e_g.get('bound_test_E'))))#-constants_test)
                        bound_test_sup.append(np.mean( np.array(e_g.get('bound_test_sup'))))#-constants_test)
                        test_errors.append(np.mean(np.array(e_g.get('test_errors'))))
                        empirical_obj=np.hstack([empirical_obj,np.mean(np.array(e_g.get('val_jest')),axis=1)])
                        pac_obj=np.hstack([pac_obj,np.mean(np.array(e_g.get('val_grad')),axis=1)])

                        #hf.close()
                  #print(e_f.shape)
                  
                  
                  #log entire epochs plots - E
                  plt.figure(figsize=(12, 10))                  
                  plt.subplot(311)
                  plt.plot(np.arange(1,len(bound_train_E)+1),np.array(bound_train_E)+np.array(train_errors),linewidth=1,color='black')
                  plt.yscale("log")
                  plt.xlabel("epochs")
                  plt.ylabel("av. train. error")
                  plt.legend(['av. training error'])
                  #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')

                  plt.subplot(312)
                  plt.plot(np.arange(1,len(bound_test_E)+1),np.array(bound_test_E)+np.array(test_errors),linewidth=1,color='blue')
                  plt.yscale("log")
                  plt.xlabel("epochs")
                  plt.ylabel("av. test error")
                  plt.legend(['av. test test error'])
                  
                  plt.subplot(313)
                  plt.plot(np.arange(1,len(pac_obj)+1),pac_obj+empirical_obj,linewidth=1,color='red')
                  plt.yscale("log")
                  plt.xlabel("iterations")
                  plt.ylabel("obj function value")
                  plt.legend(['av. obj function'])
                  
                  savename = (pathF+figure_name+'_fullEpochslog_E_'+str(dimComp([inp,*arch])) +'_' +str(datasetsM[current_id]) +'_a' +str(a_val[current_id]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.png').replace("/","-")
                  plt.savefig(savename)
                  plt.close()

                  #log entire epochs plots - sup
                  plt.figure(figsize=(12, 10))                  
                  plt.subplot(311)
                  plt.plot(np.arange(1,len(bound_train_sup)+1),np.array(bound_train_sup)+np.array(train_errors),linewidth=1,color='black')
                  plt.yscale("log")
                  plt.xlabel("epochs")
                  plt.ylabel("av. train. error")
                  plt.legend(['av. training error (bound with sup)'])
                  #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')

                  plt.subplot(312)
                  plt.plot(np.arange(1,len(bound_test_sup)+1),np.array(bound_test_sup)+np.array(test_errors),linewidth=1,color='blue')
                  plt.yscale("log")
                  plt.xlabel("epochs")
                  plt.ylabel("av. test error")
                  plt.legend(['av. test test error (bound with sup)'])
                  
                  plt.subplot(313)
                  plt.plot(np.arange(1,len(pac_obj)+1),pac_obj+empirical_obj,linewidth=1,color='red')
                  plt.yscale("log")
                  plt.xlabel("iterations")
                  plt.ylabel("obj function value")
                  plt.legend(['av. obj function (bound with E)'])
                  
                  savename = pathPrior+figure_name+'_fullEpochslog_sup_'+str(dimComp([inp,*arch])) +'_' +str(datasetsM[current_id]) +'_a' +str(a_val[current_id]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.png'.replace("/","-")
                  plt.savefig(savename)
                  plt.close()

                  """
                  #log last epochs plots
                  plt.figure(figsize=(12,10))                  
                  plt.subplot(311)
                  plt.plot(np.arange(start+1,len(bound_train_E)+1),np.array(bound_train_E[start:])+np.array(train_errors[start:]),linewidth=1,color='black')
                  plt.yscale("log")
                  plt.xlabel("epochs")
                  plt.ylabel("av. train error")
                  plt.legend(['av. training error'])
                  #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')

                  plt.subplot(312)
                  plt.plot(np.arange(start+1,len(bound_test)+1),np.array(bound_test[start:])+np.array(test_errors[start:]),linewidth=1,color='blue')
                  plt.yscale("log")
                  plt.xlabel("epochs")
                  plt.ylabel("av. test error")
                  plt.legend(['av. test error'])
                  
                  plt.subplot(313)
                  plt.plot(np.arange(Nbatches*(start)+1,len(pac_obj)+1),pac_obj[Nbatches*start:]+empirical_obj[Nbatches*start:],linewidth=1,color='red')
                  plt.yscale("log")
                  plt.xlabel("iterations")
                  plt.ylabel("objective function value")
                  plt.legend(['av. obj function'])
                  #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')
                  
                  plt.savefig(pathPrior+figure_name+'_lastEpochslog_'+str(dimComp([inp,*arch])) +'_' +str(datasetsM[current_id]) +'_a' +str(a_val[current_id]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.png')
                  plt.close()
                  """
fVal.close()              
fTest.close()
hf.close()

            
              