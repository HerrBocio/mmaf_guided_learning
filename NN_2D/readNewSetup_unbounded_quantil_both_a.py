from params_a3 import *
from params_a8 import *
from matplotlib.lines import Line2D
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
os.environ["CUDA_VISIBLE_DEVICES"]='3'#,2,3' # on cuda I have to put 0
from scipy.stats import kstest as kstest
import jax
import jax.numpy as jnp
from utils import *
data_path="datasets_2D/" # local
data_path="../datasets_2D/" # server
"#/LOCAL/jasst/results/nopreT/"#'/afs/tu-chemnitz.de/project/calibration/jasminDebug/'


only_plot_best_pir = True
#
#path='/afs/tu-chemnitz.de/project/calibration/aistatsResults/s_data_old/'
#path='/afs/tu-chemnitz.de/project/calibration/s_data_new/'

bound_train=[]
train_errors=[]

bound_test=[]
test_errors=[]

file_name= day#day+'_full_relu_std'+preTlabel#full_tanh_new_setup,full_relu_new_setup
figure_name=""#"depth_relu_std"  #day+'_depth_relu_std'#+preTlabel


a_vals = [my_aval,their_aval]
epochs = [epochs_3, epochs_8]
#a_val = [8,8]
#[p]

delta=0.025
A_estimatedM=[3.840956,3.868912]#,]#,[3.840956]#
c_estimatedM=[1,1]

piRescaling_str_val = ""
piRescaling_str_train = ""
for pir in piRescaling:
     piRescaling_str_val += "& CRPS_"+str(pir)+"& RMSE_"+str(pir)
     piRescaling_str_train += "& epoch_"+str(pir)+"& Lip_"+str(pir)+"& KL_"+str(pir)+"& empError"+str(pir)

m_batches = [m_batches]
Nbatches=124 # for N=1e6 and m=1e3
Ncoords=10
Ncones_test=101

Ncrps=100



if not os.path.exists(pathT):
            os.makedirs(pathT)
            
if not os.path.exists(pathF):
            os.makedirs(pathF)

#fTest_best_pir=open(pathT+'Table_All_Comb_Best_Pir.txt','w')
#fTest_best_pir.write('Dataset & Architecture & best Pir & best Train Err. & best Iter. train & best Test Err. & best Iter test & Validation CRPS & RMSE Val & Range CRPS_Val other prior & Test CRPS & Test RMSE\n\n')
fVal = open(pathT+ "Validation_table.txt",'w')
fVal.write('Dataset & Architecture'+piRescaling_str_val+'\n\n')

fTest = open(pathT+ "Test_table.txt",'w')
fTest.write('Dataset & Architecture & Pi & KL & rho[Lip(h)] & Bound & CRPS & RMSE'+'\n\n')

fMMAF = open(pathT+"MMAF_Table.txt",'w')
fMMAF.write("Dataset & Arch & Best Pir & m & KL& rho[Lip(h)] & rho[r(h)] & bound Train & Horizon_val & CRPS_Val & RMSE_Val & Horizon_test & CRPS_Test & RMSE_ test\n\n")

fTrain = open(pathT+"Train_Table.txt",'w')
fTrain.write('Dataset & Architecture'+piRescaling_str_train+'\n\n')


for current_id in range(len(datasetsM)):
  print('\\multirow{4}{2em}{',datasetsname_short[current_id],'}& ', file=fTrain)
  print('\\multirow{4}{2em}{',datasetsname_short[current_id],'}& ', file=fVal)
  print(datasetsname_short[current_id], file=fMMAF)
  data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
  
  
  alph=.5/A_estimatedM[current_id]
  for i_aval,a_val in enumerate(a_vals):
    data_validation=data[:Ncoords,-(Ncones_test)*a_val[current_id]:-(Ncones_test-1)*a_val[current_id]]
    b_validation=data_validation[:,-1]
    print('\\multirow{2}{*}{$',a_val[current_id],'$ $\\Bigg\\{$} ', file=fTrain)
    print('\\multirow{2}{*}{$',a_val[current_id],'$ $\\Bigg\\{$} ', file=fVal)
    for i_arch,arch in enumerate((archs)):#reversed 
      if i_arch != 0:
          print('& ', file=fTrain) 
          print('& ', file=fVal) 
      Nepochs = epochs[i_aval][i_arch]   
      Epochs = range(1,Nepochs+1)
      #fJ=open(pathT+'Table_'+figure_name+''+datasetsname_short[current_id]+str(archs[i])+'.txt','w')
      #fJ.write('Pir & best Train Err. & best Iter. train & best Test Err. & best Iter test & CRPS & RMSE & rho[Lip(h)] at best Iter train. \n\n')



      mask=mask_gen(inp,arch)
      d=dimComp([inp,*archs[i_arch]])
      best_prior = 0
      best_crps_val = np.inf
      corr_crps_test = np.inf
      range_crps_val = []
      best_rmse_val = np.inf
      corr_rmse_test = np.inf

      val_prior_row_str = ""
      train_prior_row_str =" & $"+str(arch[0])+'^'+str(len(arch)-1)+"$ "

      for pir in piRescaling:
            log_pir=-jnp.log(pir)
            piScale=jnp.ones(d)*log_pir
            piParams=[jnp.zeros(d),piScale]

            print("a, pir, arch, dataset: ",a_val[current_id],pir,arch,datasetsM[current_id])

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
              if do_f_epochs:
                f_epochs=open(pathT+'Table_'+figure_name+''+datasetsname_short[current_id]+str(archs[i_arch])+'_pir'+str(pir)+'.txt','w')
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


                    if do_f_epochs:
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
              train_emp_error_best_it = np.mean(np.array(e_g.get('train_errors')))

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
                
              #print(pir, '&', np.round(best_whole_bound_train,decimals=4),'&',best_iter_train,'&', np.round(best_whole_bound_test,decimals=4), '&', best_iter_test,'&',np.round(np.mean(crps),decimals=4),'&',np.round(np.mean(np.sqrt(rmse)),decimals=4),'&',Liph_train_best_iter,file=fJ) 
              val_prior_row_str += " & "+str(np.round(np.mean(crps),decimals=4))+'&'+str(np.round(np.mean(np.sqrt(rmse)),decimals=4))
              train_prior_row_str += " & "+str(best_iter_train)+' & '+str(np.round(Liph_train_best_iter,decimals=4))+' & '+str(np.round(KL_train_best_iter, decimals=4))+' & '+str(np.round(train_emp_error_best_it, decimals=4))

              range_crps_val.append(np.mean(crps))

              if np.mean(crps)<best_crps_val:
                    best_crps_val = np.mean(crps)
                    print("updated best prior from ",best_prior)
                    best_prior = pir
                    print("...to ",best_prior)
                    best_prior_best_train_error = best_whole_bound_train
                    best_prior_iter_train = best_iter_train
                    best_prior_best_test_error = best_whole_bound_test
                    best_prior_iter_test = best_iter_test
                    best_rmse_val = np.mean(np.sqrt(rmse))
                    best_prior_emp_train_error = train_emp_error_best_it
                    best_params_bestpir = best_params_sup


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
              qs_label=['10%','20%','30%','40%','50%','60%','70%','80%','90%','95%']#print(crps)
                
              if True:#not only_plot_best_pir:
                plt.figure()

                cmap = plt.cm.PuBu  # define the colormap
                # extract all colors from the .jet map
                cmaplist = [cmap(i) for i in range(cmap.N)]
                #print(len(cmaplist))#
                  # force the first color entry to be grey
                #cmaplist[0] = (.5, .5, .5, 1.0)
                
                # create the new map
                #cmap = mpl.colors.LinearSegmentedColormap.from_list(
                #    'Custom cmap', cmaplist, cmap.N)
                #plt.figure(figsize=(16,20))                  
                
                  
                fig1, axs1 = plt.subplots(4, 2,figsize=(8,6))
                #plt.subplots_adjust(hspace=0)
                fig1.tight_layout()
              
                for i_ax, ax in enumerate(fig1.axes):
                  #ax.set_ylabel(str(i))
                  for j,el in enumerate(qs):
                    #ax.set_ylim(-1.5,1.5)
                    ax.fill_between(np.arange(1,m_e_f_test[:,:,:Ncrps].shape[-1]+1),np.quantile(m_e_f_test[i_ax,:,:Ncrps],q=el,axis=0),np.quantile(m_e_f_test[i_ax,:,:Ncrps],q=1-el,axis=0),color=cmaplist[60+15*(j+1)])#,color='teal')
                    ax.plot(np.arange(1,b_test[:,:Ncrps].shape[-1]+1),b_test[i_ax,:Ncrps],color='orange',linewidth=.5)
                    
                  #ax.annotate('average crps: '+ str(np.round(np.mean(crps[2*i],axis=0),decimals=4)) ,xy=(30,np.amax(np.quantile(m_e_f[2*i,:,:],q=.975,axis=0))-.01),fontsize='x-small')
                
                
                custom_lines = [Line2D([0], [0],marker='s', color=cmaplist[60+15*(0+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(1+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(2+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(3+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(4+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(5+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(6+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(7+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(8+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(9+1)], lw=8),
                                ]
                fig1.legend(custom_lines,reversed([el for el in qs_label]),labelspacing=1.5, ncol=1,bbox_to_anchor=(1.09, .8))#, loc="upper left")
      #plt.colorbar(cmaplist[-1:-20*(j+1):-20])
                path_spec = pathF_all + '/'+datasetsname_short[current_id]+'/'+'a='+str(a_val[current_id])+'/'+str(arch)+'/'
                plt.savefig(path_spec+'time_EF'+figure_name+str(arch) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.png',bbox_inches='tight')
                if save_eps:
                  plt.savefig(pathF+'time_EF'+figure_name+str(arch) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.eps',bbox_inches='tight')
                plt.close()

                plt.figure()
                fig2, axs2 = plt.subplots(2, 2,figsize=(12,9))
                #plt.subplots_adjust(hspace=0)
                fig2.tight_layout()

                for i_, ax in enumerate(fig2.axes):
                  #ax.set_ylabel(str(i))
                  for j,el in enumerate(qs):
                    ax.fill_between(np.arange(1,m_e_f_test.shape[0]+1),np.quantile(m_e_f_test[:,:,2*i_],q=el,axis=1),np.quantile(m_e_f_test[:,:,2*i_],q=1-el,axis=1),color=cmaplist[60+15*(j+1)])#,color='teal')
                    ax.plot(np.arange(1,b_test.shape[0]+1),b_test[:,2*i_],color='orange',linewidth=.5)
                    
                  #ax.annotate('average crps: '+ str(np.round(np.mean(crps[2*i],axis=0),decimals=4)) ,xy=(30,np.amax(np.quantile(m_e_f[2*i,:,:],q=.975,axis=0))-.01),fontsize='x-small')
                
                
                custom_lines = [Line2D([0], [0],marker='s', color=cmaplist[60+15*(0+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(1+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(2+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(3+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(4+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(5+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(6+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(7+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(8+1)], lw=8),
                                Line2D([0], [0],marker='s', color=cmaplist[60+15*(9+1)], lw=8),
                                ]
                fig2.legend(custom_lines,reversed([el for el in qs_label]),labelspacing=1.5, ncol=1,bbox_to_anchor=(1.09, .8))#, loc="upper left")
      #plt.colorbar(cmaplist[-1:-20*(j+1):-20])
                plt.savefig(pathF+'space_EF'+figure_name+str(arch) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.png',bbox_inches='tight')
                if save_eps:
                  plt.savefig(pathF+'space_EF'+figure_name+str(arch) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.eps',bbox_inches='tight')

                plt.close()
                
              #print(pir,'&',np.round(KL,decimals=4),'&',np.round(pac_test_min,decimals=4),'&' , np.round(pac_true_min,decimals=4),'&',np.round(np.mean(crps),decimals=4),'&',np.round(np.sqrt(np.mean(rmse)),decimals=4), file=fTest)    #'&',np.round(emp_risk,decimals=4)

              if pir == best_prior:
                    corr_crps_test = np.mean(crps)
                    corr_rmse_test = np.mean(rmse)
                    corr_m_e_f_test = m_e_f_test

              #########
      
      if only_plot_best_pir:
          print(datasetsname_short[current_id], a_val[current_id], arch, best_prior) # 0 wenn best_prior nie geändert wird (warum??)
          
          """
          data_test=np.array(m_g.get('data_test'))
          b_test=np.array(e_g.get('cones_test'))
          m_e_f_test=np.zeros((0,Ndraws,Ncones_test-1))
          for j in range(p*c,data_test.shape[0]-p*c):
              coord_e_f=multi_ef_test(data_test[j-p*c:j+p*c+1,-(Ncones_test-1)*a_val[current_id]-2::a_val[current_id]],best_params_bestpir[j-p*c,:,:],inp,arch,mask,Ndraws,Ncones_test-1,rng)
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
          qs_label=['10%','20%','30%','40%','50%','60%','70%','80%','90%','95%']#print(crps)
            
          """
          plt.figure()
          cmap = plt.cm.PuBu  # define the colormap
          # extract all colors from the .jet map
          cmaplist = [cmap(i) for i in range(cmap.N)]
          #print(len(cmaplist))#
            # force the first color entry to be grey
          #cmaplist[0] = (.5, .5, .5, 1.0)
          
          # create the new map
          #cmap = mpl.colors.LinearSegmentedColormap.from_list(
          #    'Custom cmap', cmaplist, cmap.N)
          #plt.figure(figsize=(16,20))                  
          
            
          fig1, axs1 = plt.subplots(4, 2,figsize=(8,6))
          #plt.subplots_adjust(hspace=0)
          fig1.tight_layout()
        
          for i_ax, ax in enumerate(fig1.axes):
            #ax.set_ylabel(str(i))
            for j,el in enumerate(qs):
              #ax.set_ylim(-1.5,1.5)
              ax.fill_between(np.arange(1,corr_m_e_f_test[:,:,:Ncrps].shape[-1]+1),np.quantile(corr_m_e_f_test[i_ax,:,:Ncrps],q=el,axis=0),np.quantile(corr_m_e_f_test[i_ax,:,:Ncrps],q=1-el,axis=0),color=cmaplist[60+15*(j+1)])#,color='teal')
              ax.plot(np.arange(1,b_test[:,:Ncrps].shape[-1]+1),b_test[i_ax,:Ncrps],color='orange',linewidth=.5)
              
            #ax.annotate('average crps: '+ str(np.round(np.mean(crps[2*i],axis=0),decimals=4)) ,xy=(30,np.amax(np.quantile(m_e_f[2*i,:,:],q=.975,axis=0))-.01),fontsize='x-small')
          
          
          custom_lines = [Line2D([0], [0],marker='s', color=cmaplist[60+15*(0+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(1+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(2+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(3+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(4+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(5+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(6+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(7+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(8+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(9+1)], lw=8),
                          ]
          fig1.legend(custom_lines,reversed([el for el in qs_label]),labelspacing=1.5, ncol=1,bbox_to_anchor=(1.09, .8))#, loc="upper left")
#plt.colorbar(cmaplist[-1:-20*(j+1):-20])
          plt.savefig(pathF+'time_EF'+figure_name+str(arch) +'_' +str(datasetsname_short[current_id])+'a_'+str(a_val[current_id])+'pir'+str(best_prior)+'.png',bbox_inches='tight')
          if save_eps:
            plt.savefig(pathF+'time_EF'+figure_name+str(arch) +'_' +str(datasetsM[current_id])+'a_'+str(a_val[current_id])+'pir'+str(best_prior)+'.eps',bbox_inches='tight')
          plt.close()

          plt.figure()
          fig2, axs2 = plt.subplots(2, 2,figsize=(12,9))
          #plt.subplots_adjust(hspace=0)
          fig2.tight_layout()

          for i_, ax in enumerate(fig2.axes):
            #ax.set_ylabel(str(i))
            for j,el in enumerate(qs):
              ax.fill_between(np.arange(1,corr_m_e_f_test.shape[0]+1),np.quantile(corr_m_e_f_test[:,:,2*i_],q=el,axis=1),np.quantile(corr_m_e_f_test[:,:,2*i_],q=1-el,axis=1),color=cmaplist[60+15*(j+1)])#,color='teal')
              ax.plot(np.arange(1,b_test.shape[0]+1),b_test[:,2*i_],color='orange',linewidth=.5)
              
            #ax.annotate('average crps: '+ str(np.round(np.mean(crps[2*i],axis=0),decimals=4)) ,xy=(30,np.amax(np.quantile(m_e_f[2*i,:,:],q=.975,axis=0))-.01),fontsize='x-small')
          
          
          custom_lines = [Line2D([0], [0],marker='s', color=cmaplist[60+15*(0+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(1+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(2+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(3+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(4+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(5+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(6+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(7+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(8+1)], lw=8),
                          Line2D([0], [0],marker='s', color=cmaplist[60+15*(9+1)], lw=8),
                          ]
          fig2.legend(custom_lines,reversed([el for el in qs_label]),labelspacing=1.5, ncol=1,bbox_to_anchor=(1.09, .8))#, loc="upper left")
#plt.colorbar(cmaplist[-1:-20*(j+1):-20])
          plt.savefig(pathF+'space_EF'+figure_name+str(arch) +'_' +str(datasetsname_short[current_id])+'a_'+str(a_val[current_id])+'pir'+str(best_prior)+'.png',bbox_inches='tight')
          if save_eps:
            plt.savefig(pathF+'space_EF'+figure_name+str(arch) +'_' +str(datasetsM[current_id])+'a_'+str(a_val[current_id])+'pir'+str(best_prior)+'.eps',bbox_inches='tight')

          plt.close()
      
      print(val_prior_row_str,' \\'+'\\'+'\n',file =fVal)#datasetsname_short[current_id],'&',arch,' '
      #print(datasetsname_short[current_id],'& ',best_prior,'& ',train_prior_row_str,' \\'+'\\'+'\n',file =fTrain)
      print(train_prior_row_str,' \\'+'\\'+'\n',file =fTrain)
      range_crps_val.sort()
      #print(range_crps_val)
      #range_other_prior_str = "["+str(range_crps_val[1])+","+str(range_crps_val[-1])+"]"

      #print(datasetsname_short[current_id],'&',arch,'&',best_prior,'&', best_prior_best_train_error,'&',best_prior_iter_train,'&',best_prior_best_test_error,'&',best_prior_iter_test,'&',best_crps_val,'&',best_rmse_val,'&',range_other_prior_str,'&',corr_crps_test,'&',corr_rmse_test, file = fTest_best_pir)  
      
      
      print("& $",arch[0],'^',len(arch)-1,'$ & $\\mathcal{N}(0,1/',best_prior,')$ &',1000,'&',np.round(KL_train_best_iter,decimals=4) ,'&',np.round(Liph_train_best_iter,decimals=4),'&',np.round(best_prior_emp_train_error,decimals=4),'&', np.round(best_prior_best_train_error,decimals=4),'&',1,'&',np.round(best_crps_val,decimals=4),'&',np.round(best_rmse_val,decimals=4),'&',100,'&',np.round(corr_crps_test,decimals=4),'&',np.round(corr_rmse_test,decimals=4),"\\"+"\\"+"\n", file = fMMAF)  

      #print("printing done for ",datasetsname_short[current_id],arch[i_arch])
    print('\n\\cmidrule{3-25}\n',file=fTrain)
    print('\n\\cmidrule{3-25}\n',file=fVal)
  print('midrule\nmidrule\n', file=fTrain)
  print('midrule\nmidrule\n', file=fVal)


            
hf.close()

            
              