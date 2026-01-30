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
#from SGEmultiEC import post
from STOUNewSetup import ffnnV,ffnnPozzo 
import jax
import jax.numpy as jnp
pathT='results/errata_corrige/natale/tables/'
pathF='results/errata_corrige/natale/figures/'

path='/LOCAL/prol/OLR_full/natale/'
#path='/afs/tu-chemnitz.de/project/calibration/OLR_full/depth_variant/'

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

pretraining=False
if pretraining:
  Epochs=[12000]#
  Nepochs=12000#len(Epochs)
  preTlabel='_preT'
else:
  Epochs=[5000] #
  Nepochs=5000#len(Epochs)
  preTlabel=''
  
day='12_18' # 01_13
file_name= day+'_relu_rescaling'+preTlabel#full_tanh_new_setup,full_relu_new_setup
figure_name=day+'_olr'+preTlabel


if not os.path.exists(pathT+day+'/'):
    os.makedirs(pathT+day+'/')
    print("tables folder created")

if not os.path.exists(pathF+day+'/'):
    os.makedirs(pathF+day+'/')
    print("figures folder created")



datasetsM= ['OLR_full']#'Gaudiamonddata1A4mln'

machine=''

h_t=[1]
a_val=[64]

archs = [[30,30,1],[100,100,1],[300,300,1]] # [[800,800,800,1]] # 
#archs= [ [15,15,15,15,15,1],[50,50,50,50,50,1],[150,150,150,150,150,1] ]


piRescaling = list(range(10,220,20))
#piRescaling=[10,30,50,70]#[1./10,1./30,1./50,1./70]#,1./40,1./50]
#piScalingLabel=['10','30','50','70']#,50,70]#,20,30]

piScalingLabel = [str(el) for el in piRescaling]

hatA=0.6975919 
hatc=4.850653 
lambda_=0.14381401

delta=1./0.025
c=int(np.floor(hatc))
m_batches=[35] #
Nbatches=1 # for N=1e6 and m=1e3
eps=3.
Ndraws=1000

Ncoords=8+2*c
inp=2*c+1
p=1

#Epochs=range(1,8000,10) #
#Nepochs=8000 #len(Epochs)
Ncones_test=19 # 546//a_val[0] #w/ validation

Ncrps=18

calibration=False
#arch_legend=['epoch1','epoch10','epoch20','epoch30','epoch40','epoch50']

slope = 0.010816065063826066 
q = -0.13547627106824212


def rescalingInv(d,slope,q,eps=0):
  return (d-q)/slope


def multi_ef(test_cones,params,inp,arch,mask,Ndraws,Ncones_test,rng):

        #test_mapped=jax.lax.map(test_slicing,(it_coord))#print('coord',x_coord)
        # print('shape',window_mapped.shape,batch)
        #test_mapped=jnp.reshape(test_mapped,(len(it_coord),a*Ncones_test))
        #print(test_mapped.shape)
        #Ac,bc=Z.get_coneJ((test_mapped),sizeData=test_mapped.shape[1])
        #print(test_cones.shape)
        w=post([params[0],params[1]],Ndraws,seed=rng)    
        #print(w.shape)
        out=jax.vmap(lambda A : ffnnV(A,[inp,*arch],mask,w))
        m_e_f = out(test_cones)
        m_e_f = m_e_f.reshape((Ndraws,Ncones_test))
        return m_e_f

def rescalingU1(d,eps=0):
  m=np.amin(d)
  M=np.amax(d)
  #p=(1-2*eps)/(M-m) 
  #q=(M*eps-(1-eps)*m)/(M-m)
  p=2/(M-m)
  q=(m+M)/(m-M)
  return d*p+q,p,q



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

    for current_id in reversed(range(len(datasetsM))):     
            for k,pir in enumerate(piScalingLabel):      
                  print(pir,arch)
                  pathPrior=pathF+datasetsM[current_id]+'/EF/'+'prior'+piScalingLabel[k]+'var/'+day+'/'+str(dim([inp,*arch]))+'/'
                  if not os.path.exists(pathPrior):
                      os.makedirs(pathPrior)
                      print("figures folder created")
                  for m in range(len(m_batches)):  
                    for Epoch in Epochs:
                        #print(Epoch)
                        hf=h5py.File(path+'prior' + piScalingLabel[k] +'var/' +file_name +str(dim([inp,*arch])) +'_' +str(datasetsM[current_id]) +'_a' +str(a_val[current_id]) +'_pir' +piScalingLabel[k]+'_m' +str(m_batches[m]) +'_Epoch_['+str(Nepochs)+'].h5','r')
                        m_g=hf.get('m'+str(m_batches[m]))
                        data_test=np.array(m_g.get('data_test'))
                        #print('data',data_test)
                        #print(m_g.keys())
                        e_g=m_g.get('epoch'+str(Epoch))
                        data_inputs=np.transpose( data_test[:,a_val[current_id]-2::a_val[current_id]]) 
                        
                        #data_inputs= np.transpose(data_test[:,34*a_val[current_id]-2::a_val[current_id])
                        
                        #print(data_inputs.shape)
                        #e_f= np.array(e_g.get('e_f'))
                        b_test=np.array(e_g.get('cones_test'))
                        print(b_test.shape)
                        params0=np.array(m_g.get('last params'))
                        #print(params0)
                        mask=mask_gen(inp,arch)
                        
                        rng=jax.random.key(Nepochs)
                        #post_map=lambda beta: post([beta[0],beta[1]],Ndraws,seed=rng)  
                        #w=jax.vmap(post_map)(params0)
                        #print(w.shape)
                        m_e_f=np.zeros((0,Ndraws,Ncones_test-1))
                        for j in range(p*c,data_inputs.shape[1]-p*c):
                           coord_e_f=multi_ef(data_inputs[:,j-p*c:j+p*c+1],params0[j-p*c,:,:],inp,arch,mask,Ndraws,Ncones_test-1,rng)
                           #print(coord_e_f.shape)
                           m_e_f=np.vstack([m_e_f,coord_e_f.reshape((1,*coord_e_f.shape))])
                        #print(m_e_f)
                        #print(np.amax(m_e_f[0,:,:],axis=0).shape)
                        m_e_f_inv=rescalingInv(m_e_f,slope,q)
                        b_test_inv=rescalingInv(b_test,slope,q)
                        crps=[]
                        for n1 in range(8):
                          crps_coord=[]
                          for n2 in range(b_test.shape[-1]):
                            crps_coord.append(crps_univ_rank(b_test_inv[n1,n2],m_e_f_inv[n1,:,n2]))
                          crps.append(crps_coord)
                        #print(np.array(crps[0]).shape)

                        ### QUANTILE
                        
                        qs= [0.025]
                      
                        #qs=[0.,.025,.05,.1,.15,.2,.25,.3,.35,.4,.45,]#np.linspace(0.05,.45,10)##np.linspace(0.05,.45,10)
                        #qs_label=['10%','20%','30%','40%','50%','60%','70%','80%','90%','95%','100%']#print(crps)
                        plt.figure()

                        cmap = plt.cm.PuBu  # define the colormap
                        # extract all colors from the .jet map
                        cmaplist = [cmap(n3) for n3 in range(cmap.N)]
                        #print(len(cmaplist))#
                         # force the first color entry to be grey
                        #cmaplist[0] = (.5, .5, .5, 1.0)
                        
                        # create the new map
                        #cmap = mpl.colors.LinearSegmentedColormap.from_list(
                        #    'Custom cmap', cmaplist, cmap.N)
                        #plt.figure(figsize=(16,20))                  
                        
                          
                        fig, axs = plt.subplots(4, 2,figsize=(8,6))
                        #plt.subplots_adjust(hspace=0)
                        fig.tight_layout()
                      
                        for n4, ax in enumerate(fig.axes):
                          #ax.set_ylabel(str(i))
                          for n5,el in enumerate(qs):
                            ax.fill_between(np.arange(1,m_e_f[:,:,:Ncrps].shape[-1]+1),np.quantile(m_e_f_inv[n4,:,:Ncrps],q=el,axis=0),np.quantile(m_e_f_inv[n4,:,:Ncrps],q=1-el,axis=0),color='teal') #,color=cmaplist[60+15*(n5+1)])#
                            ax.plot(np.arange(1,b_test[:,:Ncrps].shape[-1]+1),b_test_inv[n4,:Ncrps],color='orange',linewidth=.5)
                            
                          #ax.annotate('average crps: '+ str(np.round(np.mean(crps[2*i],axis=0),decimals=4)) ,xy=(30,np.amax(np.quantile(m_e_f[2*i,:,:],q=.975,axis=0))-.01),fontsize='x-small')
                        
                        '''
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
                                        Line2D([0], [0],marker='s', color=cmaplist[60+15*(10+1)], lw=8)
                                       ]
                        '''
                        #fig.legend(custom_lines,reversed([el for el in qs_label]),labelspacing=1.5, ncol=1,bbox_to_anchor=(1.09, .8))#, loc="upper left")
#plt.colorbar(cmaplist[-1:-20*(j+1):-20])
                        plt.savefig(pathPrior+'time_EF'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+piScalingLabel[k]+'.png',bbox_inches='tight')
                        plt.savefig(pathPrior+'time_EF'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+piScalingLabel[k]+'.eps',bbox_inches='tight')

                        plt.close()
    
                        plt.figure()
                      
                        fig, axs = plt.subplots(4, 2,figsize=(8,6))
                        #plt.subplots_adjust(hspace=0)
                        fig.tight_layout()

                        for i, ax in enumerate(fig.axes):
                          #ax.set_ylabel(str(i))
                          for j,el in enumerate(qs):
                            ax.fill_between(np.arange(1,m_e_f.shape[0]+1),np.quantile(m_e_f_inv[:,:,i],q=el,axis=1),np.quantile(m_e_f_inv[:,:,i],q=1-el,axis=1),color='teal') #,color=cmaplist[60+15*(j+1)])#
                            ax.plot(np.arange(1,b_test.shape[0]+1),b_test_inv[:,i],color='orange',linewidth=.5)
                            
                          #ax.annotate('average crps: '+ str(np.round(np.mean(crps[2*i],axis=0),decimals=4)) ,xy=(30,np.amax(np.quantile(m_e_f[2*i,:,:],q=.975,axis=0))-.01),fontsize='x-small')
                        
                        '''
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
                        '''
                        #fig.legend(custom_lines,reversed([el for el in qs_label]),labelspacing=1.5, ncol=1,bbox_to_anchor=(1.09, .8))#, loc="upper left")
#plt.colorbar(cmaplist[-1:-20*(j+1):-20])
                        plt.savefig(pathPrior+'space_EF'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.png',bbox_inches='tight')
                        plt.close()


                    
