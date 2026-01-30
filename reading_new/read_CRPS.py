import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
from scipy.stats import chisquare
from scipy.stats import randint
from tqdm import trange
from multivariateRank import get_prerank
from SGEmultiEC import error
from STOUNewSetup import ffnnV,ffnnPozzo 
import jax
import jax.numpy as jnp

pathT='results/small_nets/tables/'
pathF='results/small_nets/figures/'

path='/LOCAL/prol/small_nets/results/'
#path = '/LOCAL/prol/m_wise_epoch_wise/nobias/'
#path='/afs/tu-chemnitz.de/project/calibration/aistatsResults/s_data_old/'#s_data/results/'
#path='/afs/tu-chemnitz.de/project/calibration/s_data_new/'

if not os.path.exists(pathT):
    os.makedirs(pathT)
    print("tables folder created")

if not os.path.exists(pathF):
    os.makedirs(pathF)
    print("figures folder created")

def dim(net):
  d=0
  for i in range(len(net)-1):
    d+=(net[i]+1)*net[i+1]
  return d


datasetsM=['Gaudiamonddata1A4mln']#,'NIGdiamonddata1A4mln']

pretraining=False


if pretraining:
  Epochs=[150] #
  Nepochs=150#len(Epochs)
  preTlabel='_preT'
else:
  Epochs=[60] #
  Nepochs=60#len(Epochs)
  preTlabel=''

day='01_28'
file_name= day+day+'_full_relu_std'+preTlabel#full_tanh_new_setup,full_relu_new_setup
figure_name=day+'_full_relu_std'+preTlabel



h_t=[1]
a_val=[8,8]#,8]
p=1
c=1
inp=3
#archs=[[30,30,1],[100,100,1],[300,300,1]]#  [[800,800,800,1]] #

archs= [[10,10,1],[10,10,10,10,10,1]] # [ [15,15,15,15,15,1],[50,50,50,50,50,1],[150,150,150,150,150,1] ]


piRescaling = list(range(10,230,20))#,20,30]
m_batches=[1000] #
Nbatches=124# for N=1e6 and m=1e3
Ndraws=int(1e3)
Ncones_test=101

Ncrps=101
#Epochs=[30]#range(1,51) #
#Nepochs=30#len(Epochs)
Ncoords=8
calibration=False
eps=3.

pre_ranks=["multivariate_rank","average_rank","band_depth","mean","variance","energy_score"]
arch_x_val=[1e3,1e4,1e5] #1,1e1,1e2,
arch_legend=['epoch1','epoch10','epoch20','epoch30','epoch40','epoch50']

def multi_ef_validation(test_cones,params,inp,arch,mask,Ndraws,Ncones_test,rng):

        #test_mapped=jax.lax.map(test_slicing,(it_coord))#print('coord',x_coord)
        # print('shape',window_mapped.shape,batch)
        #test_mapped=jnp.reshape(test_mapped,(len(it_coord),a*Ncones_test))
        #print(test_mapped.shape)
        #Ac,bc=Z.get_coneJ((test_mapped),sizeData=test_mapped.shape[1])
        #print(test_cones.shape)
        w=post([params[0],params[1]],Ndraws,seed=rng)    
        #print(w.reshape((Ncones_test,*w.shape)).shape)
        out=jax.vmap(lambda A : ffnnV(A,[inp,*arch],mask,w))
        #print(test_cones.reshape((Ncones_test,inp)).shape)
        m_e_f = out(test_cones.reshape((Ncones_test,inp)))
        m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        return m_e_f

def multi_ef_test(test_cones,params,inp,arch,mask,Ndraws,Ncones_test,rng):
        #print(test_cones)
        #test_mapped=jax.lax.map(test_slicing,(it_coord))#print('coord',x_coord)
        #print('shape',test_mapped.shape,batch)
        #test_mapped=jnp.reshape(test_mapped,(len(it_coord),a*Ncones_test))
        #print(test_mapped.shape)
        #Ac,bc=Z.get_coneJ((test_mapped),sizeData=test_mapped.shape[1])
        #print(Ac.shape)
        w=post([params[0],params[1]],Ndraws,seed=rng)    
        #print(w.shape,test_cones.shape)
        out=jax.vmap(lambda A : ffnnV(A,[inp,*arch],mask,w))
        
        m_e_f = out(jnp.transpose(test_cones))
        #print(m_e_f.shape)
        m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        #print('mef',m_e_f)
        
        
        return m_e_f

def mask_gen(inp,arch):
  
    struct=[inp,*arch]
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
    return mask

# CRPS
def crps_univ_rank(y, x):

    M=x.shape[0]

    #sum(sum(np.abs(x[j] - x[k]) for j in range(M)) for k in range(M))
    #sum(np.abs(x[j] - y) for j in range(M))
    #print(x.shape)
    #print(y.shape)
    double_sum = jax.vmap(lambda beta: jax.vmap(jnp.abs)(beta-x))
    double_sum = jnp.sum(double_sum(x))
    #print(double_sum.shape)
    crps=jnp.sum(jnp.abs(x-y))/M-double_sum/(2*M**2)
    #print('crps',crps)
    return crps



def post(rhoP,num_realizations=1,seed=1):
    
    '''
    The parameters of the neural network are stored in a linear vector, to prevent memory fill.
    The random seed is already a random key here
    The vector contains the mean and the log scale
    '''
    
    sample_shape = (num_realizations,*rhoP[0].shape)#tuple(num_realizations) + rhoP[0].shape jax.random.key(seed)
    sam=jax.random.normal(seed, shape=sample_shape) * jnp.exp(rhoP[1]/2) + rhoP[0]

    return sam
    




    #test_arch=[]
for i,arch in enumerate((archs)):#reversed
    mask=mask_gen(inp,arch)
    for current_id in reversed(range(len(datasetsM))):     
            for pir in piRescaling:      
                  print(pir,arch)
                  pathPrior=pathF+datasetsM[current_id]+'/EF/'+'prior'+str(pir)+'var/'+day+'/'+str(dim([inp,*arch]))+'/'
                  if not os.path.exists(pathPrior):
                      os.makedirs(pathPrior)
                      print("figures folder created")
                  for m in range(len(m_batches)):  
                    for Epoch in Epochs:
                        #print(Epoch)
                        hf=h5py.File(path+'prior' + str(pir) +'var/' +file_name +str(dim([inp,*arch])) +'_' +str(datasetsM[current_id]) +'_a' +str(a_val) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.h5','r')
                        m_g=hf.get('m'+str(m_batches[m]))                    
                        print(hf.keys())
                        #print(m_g.keys())
                        b_g=m_g.get('min')
                        print(b_g.keys())
                        best_params=np.array(b_g.get('best params'))
                        min_error=np.array(b_g.get('min_error'))
                        min_it=np.array(b_g.get('min iteration'))
                        data_test=(np.array(m_g.get('data_test')))
                        #print(data_test[:3,-24:])
                        e_g=m_g.get('epoch'+str(min_it))
                        bound_min=np.array(e_g.get('bound_test'))
                        emp_min=np.array(e_g.get('test_errors'))
                        test_min=jnp.mean(bound_min+emp_min)
                        pac_test_min= test_min + np.log(1./0.025)/np.sqrt(Ncones_test-1) + eps**2/(2*np.sqrt(Ncones_test-1))
                        b_test=(np.array(e_g.get('cones_test')))
                        #print('btest',b_test.shape)
                        rng=jax.random.key(Nepochs)
                        
                        '''
                        for ls_i in range(Ncoords): #might opt for an alternative shard_size
                              mile_train_e,pac_train = e_mapped_train(params[ls_i],Acones_s[ls_i],bcones_s[ls_i])
                              milestone_train_error=jnp.hstack([milestone_train_error,mile_train_e])
                              bound_train = jnp.hstack([bound_train,pac_train])
                          
                              multi_e_f,A_c_e,b_c_e=jax.vmap(finalStep)(list_shards[ls_i],params[ls_i]) #save b_c_e_stacked
                              multi_e_f_stacked=jnp.vstack([multi_e_f_stacked,multi_e_f])
                              b_c_e_stacked=jnp.vstack([b_c_e_stacked,b_c_e])
                              test_e,pac_test= e_mapped_test(params[ls_i],A_c_e,b_c_e)
                              test_error=jnp.hstack([test_error,test_e])   
                              bound_test = jnp.hstack([bound_test,pac_test])  




                      
                        
                        best_params=np.zeros((0,2,dim([inp,*arch])))
                        #print(best_params.shape)
                        #fun=lambda beta: np.concatenate([params0,beta],axis=0)
                        for el in params:
                          params0=np.vstack([params0,el])
                        #print(params0.shape)
                        #print(np.array(np.log(1./pir)),params0[0])
                        #print('params shape',params0.shape)
                        mask=mask_gen(inp,arch)
                        '''
                        rng=jax.random.key(Nepochs)
                        
    
                        ###### TEST #######
    
                        m_e_f=np.zeros((0,Ndraws,Ncones_test-1))
                        #print(data_test.shape)
                        for i in range(p*c,data_test.shape[0]-p*c):
                           coord_e_f=multi_ef_test(data_test[i-p*c:i+p*c+1,-(Ncones_test-1)*a_val[current_id]-2::a_val[current_id]],best_params[i-p*c,:,:],inp,arch,mask,Ndraws,Ncones_test-1,rng)
                           #print(i,coord_e_f.shape)
                           m_e_f=np.vstack([m_e_f,coord_e_f.reshape((1,*coord_e_f.shape))])
                        #print(m_e_f.shape)
                        #print(np.amax(m_e_f[0,:,:],axis=0).shape)
                        #print(np.quantile(m_e_f[0,:,:],q=0.025,axis=0))
                          
                        #print(np.array(crps[0]).shape)
                        qs=[.025,.05,.1,.15,.2,.25,.3,.35,.4,.45]#np.linspace(0.05,.45,10)##np.linspace(0.05,.45,10)
                        qs_label=['10%','20%','30%','40%','50%','60%','70%','80%','90%','95%']#print(crps)
                          
                        
                            
                        
                    
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
                      
                        for i, ax in enumerate(fig1.axes):
                          #ax.set_ylabel(str(i))
                          for j,el in enumerate(qs):
                            #ax.set_ylim(-1.5,1.5)
                            ax.fill_between(np.arange(1,m_e_f[:,:,:Ncrps].shape[-1]+1),np.quantile(m_e_f[i,:,:Ncrps],q=el,axis=0),np.quantile(m_e_f[i,:,:Ncrps],q=1-el,axis=0),color=cmaplist[60+15*(j+1)])#,color='teal')
                            ax.plot(np.arange(1,b_test[:,:Ncrps].shape[-1]+1),b_test[i,:Ncrps],color='orange',linewidth=.5)
                            
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
                        plt.savefig(pathPrior+'time_EF'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.png',bbox_inches='tight')
                        plt.savefig(pathPrior+'time_EF'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.eps',bbox_inches='tight')
                        plt.close()

                        plt.figure()
                        fig2, axs2 = plt.subplots(2, 2,figsize=(12,9))
                        #plt.subplots_adjust(hspace=0)
                        fig2.tight_layout()

                        for i, ax in enumerate(fig2.axes):
                          #ax.set_ylabel(str(i))
                          for j,el in enumerate(qs):
                            ax.fill_between(np.arange(1,m_e_f.shape[0]+1),np.quantile(m_e_f[:,:,2*i],q=el,axis=1),np.quantile(m_e_f[:,:,2*i],q=1-el,axis=1),color=cmaplist[60+15*(j+1)])#,color='teal')
                            ax.plot(np.arange(1,b_test.shape[0]+1),b_test[:,2*i],color='orange',linewidth=.5)
                            
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
                        plt.savefig(pathPrior+'space_EF'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.png',bbox_inches='tight')
                        plt.savefig(pathPrior+'space_EF'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.eps',bbox_inches='tight')

                        plt.close()

                    
