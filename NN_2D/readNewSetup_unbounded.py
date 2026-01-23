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
os.environ["CUDA_VISIBLE_DEVICES"]='2'#,2,3' # on cuda I have to put 0
from scipy.stats import kstest as kstest
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

file_name= day#day+'_full_relu_std'+preTlabel#full_tanh_new_setup,full_relu_new_setup
figure_name=""#"depth_relu_std"  #day+'_depth_relu_std'#+preTlabel


a_val=[8,8]
#[p]

delta=0.025
A_estimatedM=[3.840956,3.868912]#,]#,[3.840956]#
c_estimatedM=[1,1]

piRescaling=list(range(10,230,20))
piRescaling_str = ""
for pir in piRescaling:
     piRescaling_str += "& CRPS_"+str(pir)+"& RMSE_"+str(pir)

m_batches = [m_batches]
Nbatches=124 # for N=1e6 and m=1e3
Ncoords=10
Ncones_test=101

Ncrps=100



if not os.path.exists(pathT):
            os.makedirs(pathT)
            
if not os.path.exists(pathF):
            os.makedirs(pathF)

fTest_best_pir=open(pathT+'Table_All_Comb_Best_Pir.txt','w')
fTest_best_pir.write('Dataset & Architecture & best Pir & best Train Err. & best Iter. train & best Test Err. & best Iter test & Validation CRPS & RMSE Val & Range CRPS_Val other prior & Test CRPS & Test RMSE\n\n')
fVal = open(pathT+ "Validation_table.txt",'w')
fVal.write('Dataset & Architecture'+piRescaling_str+'\n\n')

fTest = open(pathT+ "Test_table.txt",'w')
fTest.write('Dataset & Architecture & Pi & KL & rho[Lip(h)] & Bound & CRPS & RMSE'+'\n\n')

fMMAF = open(pathT+"MMAF_Table.txt",'w')
fMMAF.write("Dataset & Arch & Best Pir & KL& rho[Lip(h)] & bound Train & CRPS_Val & RMSE_Val & CRPS_Test & RMSE_ test")

for current_id in range(len(datasetsM)):
    
  data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
  data_validation=data[:Ncoords,-(Ncones_test)*a_val[current_id]:-(Ncones_test-1)*a_val[current_id]]
  b_validation=data_validation[:,-1]
  
  alph=.5/A_estimatedM[current_id]
      
  for i,arch in enumerate((archs)):#reversed  
        Nepochs = epochs[i]    
        Epochs = range(1,Nepochs+1)
        fJ=open(pathT+'Table_'+figure_name+''+datasetsname_short[current_id]+str(archs[i])+'.txt','w')
        fJ.write('Pir & best Train Err. & best Iter. train & best Test Err. & best Iter test & CRPS & RMSE & rho[Lip(h)] at best Iter train. \n\n')



        mask=mask_gen(inp,arch)
        d=dimComp([inp,*archs[i]])
        best_prior = 0
        best_crps_val = np.inf
        corr_crps_test = np.inf
        range_crps_val = []
        best_rmse_val = np.inf
        corr_rmse_test = np.inf

        val_prior_row_str = ""

        for pir in piRescaling:
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
                      
                      #print("KL",np.array(e_g.get('KL')))
                      KL=np.mean(np.array(e_g.get('KL')))

                      bound_test_sup.append(np.mean( np.array(e_g.get('bound_test_sup'))))#-constants_test)
                      test_errors.append(np.mean(np.array(e_g.get('test_errors'))))
                      empirical_obj=np.hstack([empirical_obj,np.mean(np.array(e_g.get('val_jest')),axis=1)])
                      pac_obj=np.hstack([pac_obj,np.mean(np.array(e_g.get('val_grad')),axis=1)])

                      Liph_train.append(np.mean(np.array(e_g.get('Lips_train'))))
                      Liph_test.append(np.mean(np.array(e_g.get('Lips_test'))))


                      print(Epoch,'&',bound_train_sup[-1]+train_errors[-1],'&',bound_test_sup[-1]+test_errors[-1],'&',KL,'&',Liph_train[-1],file=f_epochs)

                whole_bound_train = np.array(bound_train_sup)+np.array(train_errors)
                best_whole_bound_train = np.min(whole_bound_train)
                best_iter_train = np.argmin(whole_bound_train)+1

                whole_bound_test = np.array(bound_test_sup)+np.array(test_errors)
                best_whole_bound_test = np.min(whole_bound_test)
                best_iter_test = np.argmin(whole_bound_test)+1

                print("best_iter_train", best_iter_train)
                

                rng=jax.random.key(Nepochs)
                
                e_g=m_g.get('epoch'+str(best_iter_train))
                emp_risk =np.array( e_g.get('val_jest'))
                emp_risk=np.mean(emp_risk)

                b_g=m_g.get('min')
                best_params_sup=jnp.array(b_g.get('best params_sup'))
                min_it=np.array(b_g.get('min iteration_sup'))
                min_error_sup = np.array(b_g.get('min_error_sup'))

                #print("min_it = ",min_it,"; best_iter_train=",best_iter_train)
                #print("min_error_sup",min_error_sup,"best_whole_bound_train",best_whole_bound_train)


                Liph_train_best_iter = np.mean(e_g.get('Lips_train'),axis=0)
                KL_train_best_iter = np.mean(e_g.get('KL'),axis=0)
                #Liphhhhh = np.array(e_g.get('Lips_train'))
                #KLLLLLLLLL= np.array(e_g.get('KL'))
                #print("LipHHHH", Liphhhhh)
                #print("KLLLLL",KLLLLLLLLL)

                safe_mean=lambda beta: beta/best_params_sup.shape[0]

                m_e_f_validation=np.zeros((0,Ndraws,1))
                for n1 in range(p*c,data_validation.shape[0]-p*c):
                    coord_e_f=multi_ef_validation(data_validation[n1-p*c:n1+p*c+1,-2],best_params_sup[n1-p*c,:,:],inp,arch,mask,Ndraws,1,rng)
                    m_e_f_validation=np.vstack([m_e_f_validation,coord_e_f.reshape((1,*coord_e_f.shape))])
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
                val_prior_row_str += "& "+str(np.round(np.mean(crps),decimals=4))+'&'+str(np.round(np.mean(np.sqrt(rmse)),decimals=4))

                range_crps_val.append(np.mean(crps))

                if np.mean(crps)<best_crps_val:
                     best_crps_val = np.mean(crps)
                     best_prior = pir
                     best_prior_best_train_error = best_whole_bound_train
                     best_prior_iter_train = best_iter_train
                     best_prior_best_test_error = best_whole_bound_test
                     best_prior_iter_test = best_iter_test
                     best_rmse_val = np.mean(np.sqrt(rmse))


                ########

                data_test=np.array(m_g.get('data_test'))
                b_test=np.array(e_g.get('cones_test'))
                m_e_f_test=np.zeros((0,Ndraws,Ncones_test-1))
                for j in range(p*c,data_test.shape[0]-p*c):
                    coord_e_f=multi_ef_test(data_test[j-p*c:j+p*c+1,-(Ncones_test-1)*a_val[current_id]-2::a_val[current_id]],best_params_sup[j-p*c,:,:],inp,arch,mask,Ndraws,Ncones_test-1,rng)
                    m_e_f_test=np.vstack([m_e_f_test,coord_e_f.reshape((1,*coord_e_f.shape))])
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
                  
                qs=[.025,.05,.1,.15,.2,.25,.3,.35,.4,.45]#np.linspace(0.05,.45,10)##np.linspace(0.05,.45,10)
                qs_label=['5%','10%','20%','30%','40%','50%','60%','70%','80%','90%']#print(crps)
                  
                #print(pir,'&',np.round(KL,decimals=4),'&',np.round(pac_test_min,decimals=4),'&' , np.round(pac_true_min,decimals=4),'&',np.round(np.mean(crps),decimals=4),'&',np.round(np.sqrt(np.mean(rmse)),decimals=4), file=fTest)    #'&',np.round(emp_risk,decimals=4)

                if pir == best_prior:
                     corr_crps_test = np.mean(crps)
                     corr_rmse_test = np.mean(rmse)

                #########

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
              
              savename = pathF+datasetsname_short[current_id]+ 'prior' + str(pir)+folder_day+str(arch)+figure_name +'_a' +str(a_val[current_id]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.png'.replace("/","-")
              plt.savefig(savename)
              plt.close()

        
        print(datasetsname_short[current_id],'&',arch,' ',val_prior_row_str,file =fVal)
        range_crps_val.sort()
        #print(range_crps_val)
        range_other_prior_str = "["+str(range_crps_val[1])+","+str(range_crps_val[-1])+"]"

        print(datasetsname_short[current_id],'&',arch,'&',best_prior,'&', best_prior_best_train_error,'&',best_prior_iter_train,'&',best_prior_best_test_error,'&',best_prior_iter_test,'&',best_crps_val,'&',best_rmse_val,'&',range_other_prior_str,'&',corr_crps_test,'&',corr_rmse_test, file = fTest_best_pir)  
        
        
        print(datasetsname_short[current_id],'&',arch,'&',best_prior,'&',1000,'&',np.round(KL_train_best_iter,decimals=4) ,'&',np.round(Liph_train_best_iter,decimals=4),'&', np.round(best_prior_best_train_error,decimals=4),'&',1,'&',np.round(best_crps_val,decimals=4),'&',np.round(best_rmse_val,decimals=4),'&',np.round(corr_crps_test,decimals=4),'&',np.round(corr_rmse_test,decimals=4), file = fMMAF)  


          
fJ.close() 
f_epochs.close()             
hf.close()

            
              