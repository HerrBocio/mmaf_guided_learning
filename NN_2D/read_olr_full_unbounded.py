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
import netCDF4 as nc

from SGE import error,truePAC
#from STOU import ffnnV,ffnn_
import jax
import jax.numpy as jnp
from utils import *
#from script_olr_full import linear_detrending

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'
#from scipy.stats import kstest as kstest
#from scipy.stats import chisquare
#from scipy.stats import randint
from tqdm import trange
from multivariateRank import get_prerank

pathT='results/OLR_full/tables/depth_variant/'
pathF='results/OLR_full/figures/depth_variant/'
#path='/LOCAL/prol/OLR_full/depth_variant/'


#path='/afs/tu-chemnitz.de/project/calibration/aistatsResults/OLR/'

path='/afs/tu-chemnitz.de/project/calibration/OLR_full/depth_variant/'


def dim(net):
  d=0
  for i in range(len(net)-1):
    d+=(net[i]+1)*net[i+1]
  return d

def crps_univ_rank(y, x):

    M=x.shape[0]
    crps=sum(np.linalg.norm(x[j] - y,ord=1) for j in range(M))/M - sum(sum(np.linalg.norm(x[j] - x[k],ord=1) for j in range(M)) for k in range(M))/(2*M**2)
    return crps

def crps_univ_rank_mapped(y, x):

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

def rmse_univ_wind(y,x):

    return (jnp.mean(x)-y)**2

def rmse_univ(y,x):
    #print('x shape',x.shape)
    return jnp.mean((x-y)**2)



def mae_univ(y,x):
    M=x.shape[0]
    #print('value',x)
    #print('val',jnp.sum(jnp.abs(x-y)))
    return jnp.abs(jnp.mean(x)-y)

def add_univ(y,x):
    M=x.shape[0]
    double_sum = jax.vmap(lambda beta: jax.vmap(jnp.abs)(beta-x))
    double_sum = jnp.sum(double_sum(x))
  
    return -double_sum//(2*M**2)



def chi2_diag_gaussians(piParams,rhoParams):
    """
    mu_p, mu_q: arrays shape (d,)
    var_p, var_q: arrays shape (d,) variances (>=0)
    returns: scalar chi^2(P||Q) (float or np.inf)
    """
    mu_p = jnp.exp(rhoParams[0])
    var_p = jnp.exp(rhoParams[1])
    mu_q = jnp.exp(piParams[0])
    var_q = jnp.exp(piParams[1])
    mu_p = jnp.array(mu_p)
    mu_q = jnp.array(mu_q)
    var_p = jnp.array(var_p)
    var_q = jnp.array(var_q)
    if not (mu_p.shape == mu_q.shape == var_p.shape == var_q.shape):
        raise ValueError("Shapes must match")
    #check existence condition
    #print('pos. def. check',jnp.prod(2*var_q >= var_p))
    #   return jnp.inf
    delta2 = (mu_p - mu_q)**2
    denom = var_p * (2*var_q - var_p)
    Ii = var_q / jnp.sqrt(denom) * jnp.exp(delta2 / (2*var_q - var_p))
    prod = jnp.prod(Ii)
    return prod 


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
        #print(m_e_f)
        return m_e_f

def multi_ef_test(test_cones,params,inp,arch,mask,Ndraws,Ncones_test,rng):

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
        #print('mef',m_e_f.shape)
        
        
        return m_e_f


def mask_gen(inp,arch):
  
    struct=[inp,*arch]
    mask=[0]
    s=0  
    for el in range(len(struct)-1):
       s+= (struct[el]+1)*struct[el+1]
       mask.append(s)
    return mask

def post(rhoP,num_realizations=1,seed=1):
    
    '''
    The parameters of the neural network are stored in a linear vector, to prevent memory fill.
    The random seed is already a random key here
    The vector contains the mean and the log scale
    '''
    
    sample_shape = (num_realizations,*rhoP[0].shape)#tuple(num_realizations) + rhoP[0].shape jax.random.key(seed)
    sam=jax.random.normal(seed, shape=sample_shape) * jnp.exp(rhoP[1]/2) + rhoP[0]

    return sam
    

file_name= day+'_depth_rescaling'+preTlabel#full_tanh_new_setup,full_relu_new_setup
figure_name=day+'_olr_depth_rescaling'+preTlabel


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

#archs = [[30,30,1],[100,100,1],[300,300,1]] # [[800,800,800,1]]##[[800,800,800,1]]# [30,30,1],[100,100,1],
archs= [ [15,15,15,15,15,1],[50,50,50,50,50,1],[150,150,150,150,150,1] ]


piRescaling = list(range(10,220,20))
#piRescaling=[10,30,50,70]#[1./10,1./30,1./50,1./70]#,1./40,1./50]
#piScalingLabel=['10','30','50','70']#,50,70]#,20,30]




delta=1./0.025

m_batches=[60] #
Nbatches=1 # for N=1e6 and m=1e3
eps=3.
Ndraws=1000

p=1

#Epochs=range(1,8000,10) #
#Nepochs=8000 #len(Epochs)
Ncones_test=32 # 546//a_val[0] #w/ validation

Ncrps=100

calibration=False


constants_train=2*np.log(1./0.025)/np.sqrt(m_batches[0]) + eps**2/(2*np.sqrt(m_batches[0]))

constants_test=2*np.log(1./0.025)/np.sqrt(Ncones_test-1) + eps**2/(2*np.sqrt(Ncones_test-1))

constants_test_t=np.log(1./0.025)/np.sqrt(Ncones_test-1) + eps**2/(2*np.sqrt(Ncones_test-1))

#print(constants)

start=31-1
pre_ranks=["multivariate_rank","average_rank","band_depth","mean","variance","energy_score"]
arch_x_val=[1e4,1e5] #1,1e1,1e2,
#arch_legend=['epoch1','epoch10','epoch20','epoch30','epoch40','epoch50']


file_path = '/afs/tu-chemnitz.de/project/calibration/OLR_full.nc'#Almut_plusFuture.nc'

olr = nc.Dataset(file_path, mode="r").variables
#print(olr)
olr=olr['olra'][:,:,:]
#print(olr.shape)
std_fill=-9.96921e+36

olr = np.transpose(np.mean(olr.filled(fill_value=min(-100,np.amin(olr))),axis=1))#np.nanmean(olr)
print('f')
print('shape',olr.shape,np.amax(olr),np.amin(olr))

a_t = 3.290665e-09 
a_s = -4.848268e-05 
b   = -36.211697

#detrending
#a_t,a_s,b=linear_detrending(olr[:,:])

print(a_t,a_s,b)

s_trend = lambda s: jnp.ones(olr.shape[-1])*s

s_trend = jax.vmap(s_trend)(jnp.arange(olr.shape[0]))

t_trend = lambda t: jnp.ones(olr.shape[0])*t

t_trend = jnp.transpose(jax.vmap(t_trend)(jnp.arange(olr.shape[-1])))

print('mesh', s_trend.shape,t_trend.shape)
                        
trend = a_t*t_trend + a_s*s_trend + b

print('trend',trend.shape)

mean=np.mean(olr)

olr_detrended= olr-trend


print(olr_detrended.shape)
  

#olr_detrended=np.array(list(olr_detrended))

print(olr_detrended[:,0].max(),olr_detrended[:,0].min())

#Y=np.delete(olr_detrended,bad,axis=0)

Y=np.array(list(olr_detrended))
nrY, ncY = Y.shape
print('new dataset',nrY,ncY)


#parameters estimation
c=1
dt=2
dy=10


ndata = nrY * ncY

s1 = np.nansum(Y)
s2 = np.nansum(Y**2)
s3 = np.nansum(Y**3)
s4 = np.nansum(Y**4)

k2 = (1 / (ndata * (ndata - 1))) * (ndata * s2 - s1**2)

d01 = Y.copy()

print(d01.shape)

d01[:, 0:ncY-dt] = d01[:, dt:ncY]
d01[:, ncY-dt:] = np.nan
g01 = np.nanmean((Y - d01)**2) / k2

d10 = Y.copy()
d10[0:nrY-dy, :] = d10[dy:nrY, :]
d10[nrY-dy:, :] = np.nan
g10 = np.nanmean((Y - d10)**2) / k2

print('gg',g01,g10)

hatA = -np.log(1 - g01 / 2) / ( dt)
hatc = -(hatA * dy) / np.log(1 - g10 / 2)

lambda_=hatA * np.minimum(2.0, hatc) / (2*hatc)

print('hat',hatA,hatc,lambda_)



c=int(np.floor(hatc))
print('c',c)
alph=jnp.sqrt(c)*.5/1.6693408636340383

Ncoords=8+2*c
inp=2*c+1

data,slope,q=rescalingU1(np.array(list(olr_detrended)))
print('dati',slope,q)
data_validation=data[:Ncoords,-(Ncones_test)*a_val[0]:-(Ncones_test-1)*a_val[0]]
b_validation=data_validation[:,-1]

b_validation_inv=rescalingInv(b_validation,slope,q)
#print('data',b_validation_inv)


for i,arch in enumerate(archs):#reversed 

    fTrain=open(pathT+day+'/Table_validation_'+figure_name+'_'+str(dim([inp,*archs[i]]))+'_'+machine+'.txt','w')
    fTrain.write('CRPS & RMSE\n\n')
                    
    
    fTest=open(pathT+day+'/Table_TestTrain_'+figure_name+'_'+str(dim([inp,*archs[i]]))+'_'+machine+'.txt','w')
    fTest.write('pac lin &  Test Err. & CRPS & RMSE & min it & emp.risk Train & KL train & Chi2 & true pac\n\n')
        

    #test_arch=[]
    print('dataset\n',datasetsM[0],file=fTrain)
    print('dataset \n',datasetsM[0],file=fTest)
    mask=mask_gen(inp,arch)     
    d=dim([inp,*archs[i]])
    

    for pir in piRescaling:

          #print(piScale.shape)
          print('pir', pir,'\n',file=fTrain)
          print('pir', pir,'\n',file=fTest)
          #fTrain.write('Tr.Err. & CRPS & RMSE & min_it \n\n')
          #fTest.write('Test Err. & CRPS & RMSE \n\n')
      
          print('pir',pir)
          log_pir=-jnp.log(pir)
          print(log_pir)
          piScale=jnp.ones(d)*log_pir
          if pretraining:
            
            init_scale=0.0016
            init_log_scale=jnp.log(init_scale)

            initParams = [jnp.zeros(d),jnp.ones(d)*init_log_scale] 
          
            
            piMean= prior(initParams)
            piParams=[piMean[0],piScale]
          else:
            piParams=[jnp.zeros(d),piScale]
          
          Lip=LipC(piParams,d,mask,[inp,*arch])
          print('lip', Lip)
    
          #print(piParams)
          print('pir', pir,'\n',file=fTrain)
          print('pir', pir,'\n',file=fTest)
      
          print(pir,arch)
          mask=mask_gen(inp,arch)
          pathPrior=pathF+datasetsM[0]+'/'+'prior'+str(pir)+'var/'+day+'/'+str(dim([inp,*arch])) +'/'
          if not os.path.exists(pathPrior):
              os.makedirs(pathPrior)
              print("figures folder created")
  
          empirical_obj=np.array([])
          pac_obj=np.array([])
      
          bound_train=[]
          train_errors=[]
  
          bound_test=[]
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
            hf=h5py.File(path+'prior' + str(pir) +'var/' +file_name +str(dim([inp,*arch])) +'_' +str(datasetsM[0]) +'_a' +str(a_val[0]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.h5','r')
            m_g=hf.get('m'+str(m_batches[m]))                    
            #print(hf.keys())
            #print(m_g.keys())
            b_g=m_g.get('min')
            #print(b_g.keys())
            best_params=np.array(b_g.get('best params'))
            min_error=np.array(b_g.get('min_error'))-constants_train
            min_it=np.array(b_g.get('min iteration'))
            data_test=np.array(m_g.get('data_test'))
            print(best_params.shape)
            e_g=m_g.get('epoch'+str(min_it))
            bound_min=np.array(e_g.get('bound_test'))-constants_test
            emp_min_test=np.array(e_g.get('test_errors'))
            emp_min_train=np.array(e_g.get('train_errors'))
            test_min=jnp.mean(bound_min+emp_min_test)
            pac_test_min= test_min + constants_test_t
            b_test=np.array(e_g.get('cones_test'))
            #print('bbb',b_test.shape)
            rng=jax.random.key(Nepochs)
            print('bp',best_params.shape)

            e_g=m_g.get('epoch'+str(min_it))
                  
            emp_risk =np.array( e_g.get('val_jest'))
            emp_risk=np.mean(emp_risk)
            kl_mapped=lambda beta: KLdiag_from_log_scale(piParams,beta,d)
            
            KL=jax.vmap(kl_mapped)(best_params)
            KL=jnp.mean(KL)
            print('kl ',KL)
            chi2_gau_map=lambda beta: chi2_diag_gaussians(beta,piParams)
            ChiSq=jax.vmap(chi2_gau_map)(best_params)
            Chi2=jnp.mean(ChiSq)
            print('chi2 ',Chi2)
            
            truePacFunTest=jax.vmap(lambda beta,gamma: truePAC(beta,piParams,eps,Lip,delta,inp,Ncones_test,alph,p,d,arch,gamma),in_axes=(0,0))
            true_boundTest=truePacFunTest(best_params,ChiSq)

            truePacFunTrain=jax.vmap(lambda beta,gamma: truePAC(beta,piParams,eps,Lip,delta,inp,m_batches[m],alph,p,d,arch,gamma),in_axes=(0,0))
            true_boundTrain=truePacFunTrain(best_params,ChiSq)
            
            safe_mean=lambda beta: beta/best_params.shape[0]
            chi_mean=jax.vmap(safe_mean)(ChiSq)
            Chi2=jnp.sum(chi_mean)
            print('chi2',Chi2)
            #print(emp_min + true_bound_Train)
            pac_true_min_test =jnp.mean( true_boundTest+emp_min_test)
            pac_true_min_train= jnp.mean(true_boundTrain+emp_min_train)
            print('true pac',pac_true_min_test,pac_true_min_train)  
            ##### validation #####
            
             
            m_e_f_validation=np.zeros((0,Ndraws,1))
            #print(data_validation.shape)
            for n1 in range(p*c,data_validation.shape[0]-p*c):
               coord_e_f=multi_ef_validation(data_validation[n1-p*c:n1+p*c+1,-2],best_params[n1-p*c,:,:],inp,arch,mask,Ndraws,1,rng)
               #print(i,coord_e_f.shape)
               m_e_f_validation=np.vstack([m_e_f_validation,coord_e_f.reshape((1,*coord_e_f.shape))])
            #print('pre',m_e_f_validation)
            m_e_f_validation_inv=rescalingInv(m_e_f_validation,slope,q)
            #print('post',m_e_f_validation_inv)
            crps=[]
            rmse=[]
            #mae=[]
            #add=[]
            crps_coord=[]
            rmse_coord=[]
            for n1 in range(8):
              
              #mae_coord=[]
              #add_coord=[]
              #for j in range(b_test.shape[-1]):
              crps_coord.append(crps_univ_rank_mapped(b_validation_inv[n1],m_e_f_validation_inv[n1,:,0]))
              
              rmse_coord.append(rmse_univ(b_validation_inv[n1],m_e_f_validation_inv[n1,:,0]))
              #mae_coord.append(mae_univ(b_validation_inv[n1],m_e_f_validation_inv[n1,:,0]))
              #mae.append(mae_coord)
              #add_coord.append(add_univ(b_validation_inv[n1],m_e_f_validation_inv[n1,:,0]))
              #add.append(add_coord)
            #print(np.array(crps[0]).shape)
            crps.append(crps_coord)
            rmse.append(rmse_coord)
              
            qs=[.025,.05,.1,.15,.2,.25,.3,.35,.4,.45]#np.linspace(0.05,.45,10)##np.linspace(0.05,.45,10)
            qs_label=['5%','10%','20%','30%','40%','50%','60%','70%','80%','90%']#print(crps)
            print('\t\tCRPS',np.array(crps).shape)
            print(np.round(np.mean(crps),decimals=4),'&',np.round(np.mean(np.sqrt(rmse)),decimals=4), '& ',file=fTrain)    
            
            #,np.round(min_error,decimals=4),'&', np.round(pac_true_min_train,decimals=4)np.round(np.mean(crps),decimals=4),'&',np.round(np.mean(np.sqrt(rmse)),decimals=4)
            
            ###### TEST #######
  
            m_e_f_test=np.zeros((0,Ndraws,Ncones_test-1))
            #print(data_test.shape)
            for i in range(p*c,data_test.shape[0]-p*c):
               coord_e_f=multi_ef_test(data_test[i-p*c:i+p*c+1,-(Ncones_test-1)*a_val[0]-2::a_val[0]],best_params[i-p*c,:,:],inp,arch,mask,Ndraws,Ncones_test-1,rng)
               #print(i,coord_e_f.shape)
               m_e_f_test=np.vstack([m_e_f_test,coord_e_f.reshape((1,*coord_e_f.shape))])
            
            m_e_f_test_inv=rescalingInv(m_e_f_test,slope,q)
            #print(m_e_f_test_inv)
            #print(np.amax(m_e_f[0,:,:],axis=0).shape)
            crps=[]
            rmse=[]
            for n1 in range(8):
              crps_coord=[]
              rmse_coord=[]
              for n2 in range(b_test[:,:Ncrps].shape[-1]):
                crps_coord.append(crps_univ_rank_mapped(b_test[n1,n2],m_e_f_test[n1,:,n2]))
                rmse_coord.append(rmse_univ(b_test[n1,n2],m_e_f_test[n1,:,n2]))
                #print(rmse_univ(b_test[n1,n2],m_e_f_test[n1,:,n2]).shape)
              rmse.append(np.sqrt(np.mean(rmse_coord)))
              crps.append(crps_coord) 
            print(np.array(rmse))
            qs=[.025,.05,.1,.15,.2,.25,.3,.35,.4,.45]#np.linspace(0.05,.45,10)##np.linspace(0.05,.45,10)
            qs_label=['5%','10%','20%','30%','40%','50%','60%','70%','80%','90%']#print(crps)
              
            
            print(np.round(KL,decimals=4),'&',np.round(Chi2,decimals=4),'&' , np.round(pac_test_min,decimals=4),'&',np.round(pac_true_min_test,decimals=4),'&',np.round(np.mean(crps),decimals=4),'&',np.round(np.mean(rmse),decimals=4),'&',min_it, file=fTest) 
  
  
            for Epoch in Epochs:
                #print(Epoch,dim([inp,*arch]),str(Nepochs))
                #print(path+'prior' + piScalingLabel[h] +'var/' +file_name +str(dim([inp,*arch])) +'_' +str(datasetsM[0]) +'_a' +str(a_val[0]) +'_pir' +piScalingLabel[k] +'_m' 
                e_g=m_g.get('epoch'+str(Epoch))
                #print(Epoch,e_g.keys())
                #e_f= np.array(e_g.get('e_f'))
                #data_test=np.array(e_g.get('cones_test'))
                #print(data_test[:,0])
                #e_f_inv= np.array(e_g.get('e_f_inv'))
                #print(np.mean( np.array(e_g.get('bound_train')))-constants,np.mean( np.array(e_g.get('bound_train'))))
                bound_train.append(np.mean( np.array(e_g.get('bound_train')))-constants_train)
                train_errors.append( np.mean(np.array(e_g.get('train_errors'))))
                #params=np.array(e_g.get('params'))
                #arams=jnp.
                #print(params[0].shape)
                bound_test.append(np.mean( np.array(e_g.get('bound_test')))-constants_test)
                test_errors.append(np.mean(np.array(e_g.get('test_errors'))))
                empirical_obj=np.hstack([empirical_obj,np.mean(np.array(e_g.get('val_jest')),axis=1)])
                pac_obj=np.hstack([pac_obj,np.mean(np.array(e_g.get('val_grad')),axis=1)])
  
          
          #log entire epochs plots
          plt.figure(figsize=(12, 10))                  
          plt.subplot(311)
          plt.plot(np.arange(1,len(bound_train)+1),np.array(bound_train)+np.array(train_errors),linewidth=1,color='black')
          plt.yscale("log")
          plt.xlabel("epochs")
          plt.ylabel("KL comp.")
          plt.legend(['av. training error'])
          #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')
  
          plt.subplot(312)
          plt.plot(np.arange(1,len(bound_test)+1),np.array(bound_test)+np.array(test_errors),linewidth=1,color='blue')
          plt.yscale("log")
          plt.xlabel("epochs")
          plt.ylabel("KL comp.")
          plt.legend(['av. test test error'])
          
      
          plt.subplot(313)
          plt.plot(np.arange(1,len(pac_obj)+1),pac_obj+empirical_obj,linewidth=1,color='red')
          plt.yscale("log")
          plt.xlabel("iterations")
          plt.ylabel("obj function value")
          plt.legend(['av. obj function'])
          
          
          plt.savefig(pathPrior+figure_name+'_fullEpochslog_'+str(dim([inp,*arch])) +'_' +str(datasetsM[0]) +'_a' +str(a_val[0]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.png')
          plt.close()
  
  
          #log last epochs plots
          plt.figure(figsize=(12,10))                  
          plt.subplot(311)
          plt.plot(np.arange(start+1,len(bound_train)+1),np.array(bound_train[start:])+np.array(train_errors[start:]),linewidth=1,color='black')
          plt.yscale("log")
          plt.xlabel("epochs")
          plt.ylabel("KL comp.")
          plt.legend(['av. training error'])
          #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')
  
          plt.subplot(312)
          plt.plot(np.arange(start+1,len(bound_test)+1),np.array(bound_test[start:])+np.array(test_errors[start:]),linewidth=1,color='blue')
          plt.yscale("log")
          plt.xlabel("epochs")
          plt.ylabel("KL comp.")
          plt.legend(['av. test error'])
          
          plt.subplot(313)
          plt.plot(np.arange(Nbatches*(start)+1,len(pac_obj)+1),pac_obj[Nbatches*start:]+empirical_obj[Nbatches*start:],linewidth=1,color='red')
          plt.yscale("log")
          plt.xlabel("iterations")
          plt.ylabel("objective function value")
          plt.legend(['av. obj function'])
          #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')
          
          plt.savefig(pathPrior+figure_name+'_lastEpochslog_'+str(dim([inp,*arch])) +'_' +str(datasetsM[0]) +'_a' +str(a_val[0]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.png')
          plt.close()
  
  
          #rank histogram plots ["multivariate_rank","average_rank","band_depth","mean","variance","energy_score"]
          plt.figure(figsize=(12,10))                  
          plt.subplot(321)
          plt.stairs(rank_mvr_stacked,fill=True)
          plt.legend(['multivariate rank'])
          #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')
  
          plt.subplot(322)
          plt.stairs(rank_avr_stacked,fill=True)
          plt.legend(['average rank'])
  
          plt.subplot(323)
          plt.stairs(rank_bd_stacked,fill=True)
          plt.legend(['band depth'])
          #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')
  
          plt.subplot(324)
          plt.stairs(rank_mean_stacked,fill=True)
          plt.legend(['mean'])
          
          plt.subplot(325)
          plt.stairs(rank_var_stacked,fill=True)
          plt.legend(['variance'])
          #plt.annotate('bound validation set '+str(bou) ,xy=(0,-0.9),fontsize='x-small')
          
          plt.subplot(326)
          plt.stairs(rank_es_stacked,fill=True)
          plt.savefig(pathPrior+figure_name+'_rankHistograms_'+str(dim([inp,*arch])) +'_' +str(datasetsM[0]) +'_a' +str(a_val[0]) +'_pir' +str(pir) +'_m' +str(m_batches[m]) +'_Epoch_'+str(Nepochs)+'.png')
          plt.legend(['energy_score'])      
          plt.close()
            
            
            
hf.close()
                     
       



            



              
