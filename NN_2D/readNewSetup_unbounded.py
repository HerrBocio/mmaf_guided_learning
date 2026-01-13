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
data_path="../datasets_2D/" # server
"#/LOCAL/jasst/results/nopreT/"#'/afs/tu-chemnitz.de/project/calibration/jasminDebug/'

#
#path='/afs/tu-chemnitz.de/project/calibration/aistatsResults/s_data_old/'
#path='/afs/tu-chemnitz.de/project/calibration/s_data_new/'

bound_train=[]
train_errors=[]

bound_test=[]
test_errors=[]


Epochs=range(1,epochs_nopreT+1) 
Nepochs=epochs_nopreT
preTlabel=''

file_name= day#day+'_full_relu_std'+preTlabel#full_tanh_new_setup,full_relu_new_setup
figure_name=""#"depth_relu_std"  #day+'_depth_relu_std'#+preTlabel


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

#pre_ranks=["multivariate_rank","average_rank","band_depth","mean","variance","energy_score"]
#arch_x_val=[1e4,1e5] #1,1e1,1e2,
#arch_legend=['epoch1','epoch10','epoch20','epoch30','epoch40','epoch50']

if not os.path.exists(pathT):
            os.makedirs(pathT)
            
if not os.path.exists(pathF):
            os.makedirs(pathF)

fTest_best_pir=open(pathT+'Table_Test_All_Pir.txt','w')
fTest_best_pir.write('Dataset & Architecture & best Pir & Test CRPS\n\n')

for current_id in reversed(range(len(datasetsM))):
    
  data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
  data_validation=data[:Ncoords,-(Ncones_test)*a_val[current_id]:-(Ncones_test-1)*a_val[current_id]]
  #print(data_validation)
  b_validation=data_validation[:,-1]
  
  alph=.5/A_estimatedM[current_id]
      
  for i,arch in enumerate((archs)):#reversed      
        fJ=open(pathT+'Table_'+figure_name+''+datasetsname_short[current_id]+str(archs[i])+'.txt','w')
        fJ.write('Pir & best Train Err. & best Iter. train & best Test Err. & best Iter test & CRPS & RMSE & rho[Lip(h)] at best Iter train. \n\n')

        fTest=open(pathT+'Table_Test_'+figure_name+''+datasetsname_short[current_id]+str(archs[i])+'.txt','w')


        mask=mask_gen(inp,arch)
        d=dimComp([inp,*archs[i]])
        best_prior = 0
        best_crps_val = np.inf
        corr_crps_test = np.inf

        for pir in piRescaling:
              print('pir', pir,'\n',file=fTest)
              log_pir=-jnp.log(pir)
              piScale=jnp.ones(d)*log_pir
              piParams=[jnp.zeros(d),piScale]

              print("pir, arch, dataset: ",pir,arch,datasetsM[current_id])

              empirical_obj=np.array([])
              pac_obj=np.array([])
          
              bound_train_sup=[]
              train_errors=[]

              bound_test_sup=[]
              test_errors=[]

              Liph_train = []
              Liph_test = []
              
              for m in range(len(m_batches)): 
                path_and_file_name = os.path.join(
                    path,
                    'prior' + str(pir) + 'var',
                    file_name + str(arch) + '_' +
                    str(datasetsM[current_id])[:3] + '_a' + str(a_val[current_id]) +
                    '_pir' + str(pir) + '_m' + str(m_batches[m]) +
                    '_Epoch_' + str(Nepochs) + '.h5'
                )
                try:
                  hf = h5py.File(path_and_file_name, 'r')
                except OSError as e:
                    print(path_and_file_name," FEHLER")
                    continue
                f_epochs=open(pathT+'Table_'+figure_name+''+datasetsname_short[current_id]+str(archs[i])+'_pir'+str(pir)+'.txt','w')
                f_epochs.write('Epoch & Train Err. & Test Err. & KL & rho[Lip(h)]\n\n')  

                for Epoch in Epochs:
                      m_g=hf.get('m'+str(m_batches[m])) 
                      e_g=m_g.get('epoch'+str(Epoch))
                      
                      #print(e_g.keys())
                      #e_f= np.array(e_g.get('e_f'))
                      #data_test=np.array(e_g.get('cones_test'))
                      #print(data_test[:,0])
                      #e_f_inv= np.array(e_g.get('e_f_inv'))
                      bound_train_sup.append(np.mean( np.array(e_g.get('bound_train_sup'))))#-constants_train)
                      train_errors.append( np.mean(np.array(e_g.get('train_errors'))))
                      #min_bound_train.append(np.mean( np.array(e_g.get('min bound'))))
                      
                      params=np.array(e_g.get('params_stacked'))
                      kl_mapped=lambda beta: KLdiag_from_log_scale(piParams,beta,d)#(piParams,rhoParams,NNsize):

                      KL = jax.vmap(kl_mapped)(params)
                      KL = jnp.mean(KL)
                      #arams=jnp.
                      #print(params[0].shape)
                      #print(np.array(e_g.get('best_paramsmin error')))
                      bound_test_sup.append(np.mean( np.array(e_g.get('bound_test_sup'))))#-constants_test)
                      test_errors.append(np.mean(np.array(e_g.get('test_errors'))))
                      empirical_obj=np.hstack([empirical_obj,np.mean(np.array(e_g.get('val_jest')),axis=1)])
                      pac_obj=np.hstack([pac_obj,np.mean(np.array(e_g.get('val_grad')),axis=1)])

                      Liph_train.append(np.mean(np.array(e_g.get('Lips_train'))))
                      Liph_test.append(np.mean(np.array(e_g.get('Lips_test'))))

                      print(e_g.get("params_stacked"))

                      print(Epoch,'&',bound_train_sup[-1]+train_errors[-1],'&',bound_test_sup[-1]+test_errors[-1],'&',KL,'&',Liph_train[-1],file=f_epochs)

                for elem in bound_train_sup:
                    print("bound train sup ",elem)

                for ele in bound_test_sup:
                    print("bound test sup", ele)
                
                whole_bound_train = np.array(bound_train_sup)+np.array(train_errors)
                best_whole_bound_train = np.min(whole_bound_train)
                best_iter_train = np.argmin(whole_bound_train)+1

                whole_bound_test = np.array(bound_test_sup)+np.array(test_errors)
                best_whole_bound_test = np.min(whole_bound_test)
                best_iter_test = np.argmin(whole_bound_test)+1

                print("best_iter_train", best_iter_train)
                
                for elem in whole_bound_train:
                    print("whole bound train sup ",elem)

                for ele in whole_bound_test:
                    print("whole bound test sup", ele)

                #print("!!!!!!!!!!!!!!!!!!!!!",np.min(whole_bound_train))
                #print("????????????", np.min(whole_bound_test))

                rng=jax.random.key(Nepochs)
                
                e_g=m_g.get('epoch'+str(best_iter_train))
                best_params_sup = e_g.get('params_stacked')
                emp_risk =np.array( e_g.get('val_jest'))
                emp_risk=np.mean(emp_risk)

                print("params",best_params_sup)
                print("emp_risk", emp_risk)

                Liph_train_best_iter = np.mean(e_g.get('Lips_train'),axis=0)
                print("Lip",Liph_train_best_iter)

                safe_mean=lambda beta: beta/best_params_sup.shape[0]

                m_e_f_validation=np.zeros((0,Ndraws,1))
                #print(data_validation.shape)
                for n1 in range(p*c,data_validation.shape[0]-p*c):
                    coord_e_f=multi_ef_validation(data_validation[n1-p*c:n1+p*c+1,-2],best_params_sup[n1-p*c,:,:],inp,arch,mask,Ndraws,1,rng)
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
                  
                print(pir, '&', np.round(best_whole_bound_train,decimals=4),'&',best_iter_train,'&', np.round(best_whole_bound_test,decimals=4), '&', best_iter_test,'&',np.round(np.mean(crps),decimals=4),'&',np.round(np.mean(np.sqrt(rmse)),decimals=4),'&',Liph_train_best_iter,file=fJ) 

                if np.mean(crps)<best_crps_val:
                     best_crps_val = np.mean(crps)
                     best_prior = pir

                ########

                data_test=np.array(m_g.get('data_test'))
                b_test=np.array(e_g.get('cones_test'))
                m_e_f_test=np.zeros((0,Ndraws,Ncones_test-1))
                #print(data_test.shape)
                for j in range(p*c,data_test.shape[0]-p*c):
                    coord_e_f=multi_ef_test(data_test[j-p*c:j+p*c+1,-(Ncones_test-1)*a_val[current_id]-2::a_val[current_id]],best_params_sup[j-p*c,:,:],inp,arch,mask,Ndraws,Ncones_test-1,rng)
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
                  
                print(np.round(np.mean(crps),decimals=4), file=fTest) 
                #print(np.round(KL,decimals=4),'&',np.round(pac_test_min,decimals=4),'&' , np.round(pac_true_min,decimals=4),'&',np.round(np.mean(crps),decimals=4),'&',np.round(np.sqrt(np.mean(rmse)),decimals=4),'&',min_it, file=fTest)    #'&',np.round(emp_risk,decimals=4)

                if pir == best_prior:
                     corr_crps_test = np.mean(crps)

                #########

              print(datasetsname_short[current_id],'&',arch,'&',best_prior,'&', best_crps_val, file = fTest_best_pir)

              #log entire epochs plots - sup
              plt.figure(figsize=(12, 10))                  
              plt.subplot(411)
              plt.plot(np.arange(1,len(bound_train_sup)+1),np.array(bound_train_sup)+np.array(train_errors),linewidth=1,color='black')
              plt.yscale("log")
              plt.xlabel("epochs")
              plt.ylabel("av. train. error")
              plt.legend(['av. training error (bound with sup)'])
              #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')

              plt.subplot(412)
              plt.plot(np.arange(1,len(bound_test_sup)+1),np.array(bound_test_sup)+np.array(test_errors),linewidth=1,color='blue')
              plt.yscale("log")
              plt.xlabel("epochs")
              plt.ylabel("av. test error")
              plt.legend(['av. test test error (bound with sup)'])
              
              plt.subplot(413)
              plt.plot(np.arange(1,len(pac_obj)+1),pac_obj+empirical_obj,linewidth=1,color='red')
              plt.yscale("log")
              plt.xlabel("iterations")
              plt.ylabel("obj function value")
              plt.legend(['av. obj function (using bound with E)'])

              plt.subplot(414)
              plt.plot(np.arange(1,len(Liph_train)+1),Liph_train,linewidth=1,color='red')
              #plt.yscale("log")
              plt.xlabel("iterations")
              plt.ylabel("rho[Lip(h)]")
              plt.legend(['average expected value of Lip(h)'])
              
              savename = pathF+"SUP"+datasetsname_short[current_id]+ 'prior' + str(pir)+folder_day+str(arch)+figure_name +'_a' +str(a_val[current_id]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.png'.replace("/","-")
              plt.savefig(savename)
              plt.close()

          
          
fJ.close() 
f_epochs.close()             
hf.close()

            
              