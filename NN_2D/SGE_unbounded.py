import numpy as np
from scipy.io import loadmat
from jax import grad, jit
from jax import lax
from jax import random
import jax
from jax import jit
import jax.numpy as jnp
import optax
from STOU import *
from utils import * 
from sge_utils import *
from scipy.stats import randint
from tqdm import trange,tqdm
#from optax._src import wrappers,utils
#from optax.monte_carlo import stochastic_gradient_estimators as sge
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'#,2,3'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
  
# def target_func_unbounded(piParams,rhoParams,NNsize,m):
#    return KLdiag(piParams,rhoParams,NNsize)/jnp.sqrt(m) + target_func_unbouded_sampling_from_rho() # ist KLdiag richtig oder from_logscale???

def target_func_unbounded_KL(piParams,rhoParams,NNsize,m):
   return KLdiag_from_log_scale(piParams,rhoParams,NNsize)/jnp.sqrt(m)
   

def target_func_unbouded_sampling_from_rho(A,b,realization,rhoParams,dim,arch,mask,theta, VarZtrx,m,delta):#,return_lip = False): 
  tf = empirical_risk_unbounded(A,b,realization,arch,mask)

  def pozzo(params,mask,arch):
      pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
      return pozzo
  realization=pozzo(realization,mask,arch)
  forward=lambda alpha:(ffnn_forward_pass(alpha,realization))
  hX=jax.vmap(forward,in_axes=1)(A)
  hX=hX.reshape((hX.shape[0]))
  apc = A.shape[0]
  rho_Liph = Lip_realizations_masked(realization)
  rho_Liph_sq = jnp.mean(jnp.power(rho_Liph,2))
  abs_mean = lambda beta: jnp.abs(jnp.mean(beta))
  abs_E_hX = jax.vmap(abs_mean)(hX)
  rho_abs_E_hX_Liph = jnp.mean(jnp.multiply(abs_E_hX, rho_Liph))
  rho_hXsq = jnp.mean(jnp.power(abs_E_hX,2))
  tf += apc*rho_Liph*(theta + VarZtrx/jnp.sqrt(m))
  tf += rho_Liph_sq*apc**2/(2*jnp.sqrt(m))*(VarZtrx+theta**2)
  tf += apc * theta / jnp.sqrt(m) * rho_abs_E_hX_Liph
  tf += rho_hXsq/(2*jnp.sqrt(m))
  #Erho = LipC(rhoParams,dim,mask,arch) # this is not E[rho[Lip(h)]], just an estimation with one draw
  #tf += 1/delta**2*(1/jnp.sqrt(m)*Erho**2 + 2*apc*theta*Erho)
  tf += 1/delta**2*(1/jnp.sqrt(m)*rho_Liph**2 + 2*apc*theta*rho_Liph)
  return tf

def tf_unbounded(A,b,rhoParams,dim,arch,mask,theta, VarZtrx,m,delta):
   #h = dist_sample(rhoParams,1)
   #return lambda beta: empirical_risk_unbounded(A,b,beta,arch,mask) 
   return lambda beta: target_func_unbouded_sampling_from_rho(A,b,beta,rhoParams,dim,arch,mask,theta, VarZtrx,m,delta)

def pac_unbounded(A,realizations,piParams,rhoParams,dim,arch,mask,theta,VarZtrx,m,delta):
    def pozzo(params,mask,arch):
      pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
      return pozzo
    masking = lambda alpha: pozzo(alpha,mask,arch)
    realizations=jax.vmap(masking,in_axes=0)(realizations)
    forward=lambda alpha:(ffnn_forward_pass_several_weights(alpha,realizations))
    hX=jax.vmap(forward,in_axes=1)(A) #(1000,5)
    apc = A.shape[0]
    Liphs = Lip_realizations_masked(realizations) # (5,) - passt :)
    rho_Liph = jnp.mean(Liphs)
    rho_Liph_sq = jnp.mean(jnp.power(Liphs,2))
    abs_mean = lambda beta: jnp.abs(jnp.mean(beta, axis=0))
    hX = jnp.stack(hX, axis=1)
    abs_E_hX = jax.vmap(abs_mean)(hX)
    rho_abs_E_hX_Liph = jnp.mean(jnp.multiply(abs_E_hX, Liphs))
    rho_hXsq = jnp.mean(jnp.power(abs_E_hX,2))
    tf_E = apc*rho_Liph*(theta + VarZtrx/jnp.sqrt(m))
    tf_E += rho_Liph_sq*apc**2/(2*jnp.sqrt(m))*(VarZtrx+theta**2)
    tf_E += apc * theta / jnp.sqrt(m) * rho_abs_E_hX_Liph
    tf_E += rho_hXsq/(2*jnp.sqrt(m))
    #Erho = rho_Liph #LipC(rhoParams,dim,mask,arch) # this is not E[rho[Lip(h)]], just an estimation with one draw
    #tf += 1/delta**2*(1/jnp.sqrt(m)*Erho**2 + 2*apc*theta*Erho)
    tf_sup = tf_E + theta*apc/delta * rho_Liph # rho_Liph as an estimation for sup rho[Lip(h)]
    tf_E += 1/delta**2*(1/jnp.sqrt(m)*rho_Liph**2 + 2*apc*theta*rho_Liph)
    constants_sup = theta*(1+1/delta)+jnp.log(1/delta)/jnp.sqrt(m)+VarZtrx/(2*jnp.sqrt(m))
    constants_E = constants_sup +jnp.sqrt(m)*(apc*theta/delta)**2
    bound_E = tf_E + target_func_unbounded_KL(piParams,rhoParams,dim,m) + constants_E
    bound_sup = tf_sup + target_func_unbounded_KL(piParams,rhoParams,dim,m) + constants_sup
    return [bound_E,bound_sup]

def l_empirical_risk_unbounded(A,b,arch,mask):
    #lambda function for the computation of the empirical risk (to compute the gradient over)
    return lambda beta: empirical_risk_unbounded(A,b,beta,arch,mask)

def r_empirical_risk_unbounded(A,b,arch,mask):
    #lambda function for the computation of the empirical risk ()
    return lambda beta: empirical_val_unbounded(A,b,beta,arch,mask)


def empirical_risk_unbounded(A,b,realization,arch,mask):

    '''
    Function that computes the empirical risk with respect to the absolute loss
    Input:
        A: cone inputs X_i
        b: cone outputs Y_i
        realization: samples from the newtork distribution
        arch: network layers 
        mask: network architecture structure
        dim: size of the network
        eps: truncation level of the loss function
    Output:
        eU: vector of empirical risk for each single draw from the distribution
    '''
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo
    #reshapes the network parameter according to the layer structure
    realization=pozzo(realization,mask,arch)
    #computes the loss function over all the distribution draws
    empR=lambda beta : ffnn_loss_forward_pass_unbounded(A,beta,b)
    eU=empR(realization)
    return eU

def empirical_val_unbounded(A,b,realization,arch,mask):
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo
    masking= lambda a: pozzo(a,mask,arch)
    realization=jax.vmap(masking)(realization)
    empR=lambda beta : ffnn_loss_forward_pass_unbounded(A,beta,b)
    eU=jax.vmap(empR)(realization)
    return eU

def error_unbounded(it_params,Z,A_c_e,b_c_e,dim,arch,mask,piParams,m,delta,rng,shard_size=int(5e0),N=int(1e1)):

    '''
    Function computing the pac bound for a generic dataset, for a single spatial coordinate
    Outputs the two terms separately
    '''
    #jax.debug.print("start of error_unbounded {}",A.shape)
    #lambda function, computes the value for the empirical risk
    MC_train_e = r_empirical_risk_unbounded(A_c_e,b_c_e,arch, mask)
    #m= Z.Ncones

    # N draws from the predictive distribution
    realizations = dist_sample([it_params[0],it_params[1]],num_realizations=N,seed=rng)
    realizations = realizations.reshape((N//shard_size,shard_size,dim))
    
    error=0
    pac_E = 0 
    pac_sup = 0 
    theta = Z.thetatilder
    VarZtrx = Z.truncated_covs_between_all_members_of_cone()[1][0][0]
    for el in realizations:
    #serial parallelization for each shard
      e = MC_train_e(el)
      error += jnp.sum(e)
      #jax.debug.print("realiyation shape: {}", el.shape)
      pE,psup = pac_unbounded(A_c_e,el,piParams,it_params,dim,arch,mask,theta,VarZtrx,m,delta) # A instead of A_c_e, but rethink in ways I shard
      pac_E += jnp.sum(pE)
      pac_sup += jnp.sum(psup)
    error=error/N
    pac_E=pac_E/N
    pac_sup=pac_sup/N
    jax.debug.print("error {}", error)
    jax.debug.print("pac_E {}", pac_E)
    jax.debug.print("pac_sup {}", pac_sup)
    return [error,pac_E,pac_sup]


def coordit_pretraining(Z,preT_coord,preT_d_slicing,a,c,p,arch,mask,dim,eps,init_scale=0.0016,pretraining_learning_rate=0.01): # not done for unbounded case yet

  '''
  Function performing the pretraining stagel, over a predefined training subset, for a spatial coordinate
  Outputs the vector of parameters w, the random initialization w0, the emprical risk at pretraining
  '''

  #spatial slicing of the pretraining dataset
  preT_mapped=jax.vmap(preT_d_slicing)(preT_coord)
  preT_mapped=jnp.reshape(preT_mapped,(len(preT_coord),preT_mapped.shape[-1]))

  #cone extraction
  Ac_preT,bc_preT = Z.get_coneJ(preT_mapped,sizeData=preT_mapped.shape[1],preT=True)

  #parameter initialization
  init_log_scale=jnp.log(init_scale)
  initParams = [jnp.zeros(dim),jnp.ones(dim)*init_log_scale] 

  #random draw from the initial distribution
  w0 = dist_sample(initParams,seed=jax.random.key(0))
  w=w0

  #choice of optimizer, Adam by default
  optimizer = optax.adam(pretraining_learning_rate)	 
  opt_state = optimizer.init(w)

  #the number of epochs is intended to correspond to 12k iterations, in accordance to Dziugate, Roy (2017)
  for e in range(12):
    for i in range(Z.Ncones):
      #classic optimization routine
      scorf = l_empirical_risk_unbounded(Ac_preT[:,i].reshape((Ac_preT[:,i].shape[0],1)),bc_preT[i],arch, mask, dim, eps)
      grads =jax.vmap(jax.grad(scorf))(w)
      updates, opt_state = optimizer.update(grads, opt_state)
      w = optax.apply_updates(w, updates)
  preT_val=jax.vmap(scorf)(w)
  return w,w0,preT_val



#@jit
def Optimization(file_m,Z,x_size,preT,rescaling,eps,delta,data,inp,p,c,arch,dim,Ndraws,Ncones,Ncones_test,Ncoords,shard_size,lr,slope=1,q=0,epochs=1,maxit=jnp.inf,piScaling=1):

    '''
    Performs the MMAF optimization routine 
    '''
  
    mask = mask_gen(inp,arch)

    if rescaling:
      print('rescaling between [-1,1]')
      data,pinv,qinv=rescalingU1(data)
      file_m.create_dataset('slope',data=pinv)
      file_m.create_dataset('quota',data=qinv)


    #train,validation,test split
    data_test=data[:,-(Ncones_test-1)*Z.a:] 
    data_train=data[:,:-Z.a*Ncones_test]
    data_pretraining = data_train[:,-1001*Z.a:] # extra cone for the last point 
    
    #adds the training dataset to the h5
    file_m.create_dataset('data_test',data=data_test)
  
    N=data_train.shape[1]
    x_size=data_train.shape[0]

    #creates the sharded mapping for secure parallelization
    list_windows= jnp.array([jnp.arange(element-c*p,element+c*p+1) for element in range(c*p,x_size-c*p)])
    
  
    preT_d_slicing= lambda beta: lax.dynamic_index_in_dim(data_pretraining,beta,axis=0) 
    

    # PRETRAINING stage
    pretraining_mean= jnp.zeros(dim)
    if preT:
      print('pretraining')
      pretraining = jax.vmap(lambda preT_coord: coordit_pretraining(Z,preT_coord,preT_d_slicing,Z.a,Z.c_,Z.p,[inp,*arch],mask,dim,eps))
      
      pretraining_mean,w0,preT_emp_risk = pretraining(list_windows) 
      print('saving preT_ER')
      file_m.create_dataset('pretraining ER',data=preT_emp_risk)

    ### PAC-BAYES BOUND OPTIMIZATION

  
    #data structure initialization
    coord = []
    rng = jax.random.key(0)
    output=[]
    for t in reversed(range(p)): #parallelize!!!
        coord.append(jnp.array([[v, - (t + 1)] for v in range(-c*(t + 1), c*(t + 1) + 1)]))
    coord = jnp.concat(coord, axis=0)
    coord = jnp.expand_dims(jnp.expand_dims(coord, 0), 0)
    
    bounds=[]
    pars=[]
 
    aux_in=[inp,*arch[:-1]]

    piScale=jnp.array([])
    for in_,out_ in zip(aux_in,arch):
      layer=jnp.ones((in_+1)*out_)/in_
      piScale=jnp.hstack([piScale,layer])


    if piScaling!=1:
      print('fixed setup, posterior with zero mean')
      piScale=jnp.ones(dim)*piScaling
    else: 
        print('mixed setup, post erior with zero mean')


    #network parameters initialization
    if preT:
      #if pretraining is performed, the posterior mean is set as the resulting parameters of pretraining 
      
      pretraining_fun = jax.vmap(lambda beta: jnp.vstack([beta,jnp.ones(dim)*jnp.linalg.norm(beta,ord=1)]))
      pretraining_parameters = pretraining_fun(pretraining_mean)
      params=[pretraining_parameters[el*shard_size:(el+1)*shard_size] for el in range((x_size-2*p*c)//shard_size) ]
      piParams = [w0[0,0,:],piScale]
      
    else:
      #if no pretraining is performed, the posterior mean is set as having variance equal to 1/4
      rhoParams=[jnp.zeros(dim),jnp.ones(dim)*jnp.log(.25)] #jnp.ones(dim)/2
      sharded_params=jax.vmap(lambda dummy: jnp.vstack(rhoParams))(range(shard_size))
      params= [sharded_params for el in range((x_size-2*p*c)//shard_size)]   
      piParams=[jnp.zeros(dim),piScale]
      
   
    start_learning_rate = jnp.float32(lr)

    #choice of OPTIMIZER, adam is selected by default

    optimizer = optax.adam(start_learning_rate)	
    #initial optimization stage, set accordingly to choice of pretraining
    if preT:
      opt_state_grid= [jax.vmap(optimizer.init)(el) for el in params]
    else:
      opt_state_grid = [jax.vmap(optimizer.init)(sharded_params) for el in range((x_size-2*p*c)//shard_size)]

    Acones_dummy = jnp.zeros((Ncones,inp))
    bcones_dummy = jnp.zeros(Ncones)
    
    sharded_Acones = jax.vmap(lambda dummy: jnp.vstack(Acones_dummy))(range(shard_size))
    Acones_s = [sharded_Acones for el in range((x_size-2*p*c)//shard_size)] 

    sharded_bcones = jax.vmap(lambda dummy: jnp.vstack(bcones_dummy))(range(shard_size))
    bcones_s = [sharded_bcones for el in range((x_size-2*p*c)//shard_size)] 

    
    time_windows=([jnp.arange(jnp.maximum(0,Z.N-Z.a*Z.Ncones*(element+1)),(Z.N-Z.a*Z.Ncones*element)) for element in range(Z.Nbatches)])
    t_slicing= lambda beta: lax.dynamic_index_in_dim(jnp.transpose(data_train), beta,axis=0)
    list_shards= ([jnp.array([jnp.arange(c_coord-p*c,c_coord+p*c+1) for c_coord in range(element,element+shard_size)]) for element in range(p*c,x_size-p*c,shard_size)])

    def ef_setup(it_coord,params,a,inp,arch,mask,Ndraws,Ncones_test,rng):
  
        '''
        Function that extracts the cones from the test set
        '''
      
        test_mapped=jax.lax.map(test_slicing,(it_coord))
        test_mapped=jnp.reshape(test_mapped,(len(it_coord),a*Ncones_test))
        Ac,bc=Z.get_coneJ((test_mapped),sizeData=test_mapped.shape[1])
        
        return [Ac,bc] 

    def coordit_unbounded(it_coord,it_params,opt_state,batch,Bsize,m,dim,inp,delta,rng=rng):

        '''
        Core function for the optimization: performs the gradient step and updates the parameters for each spatial coordinate at the current batch iteration
        '''
        #jax.debug.print("params {}", it_params)
        #spatial training set slicing 
        window_mapped=jax.vmap(d_slicing)(it_coord)
        window_mapped=jnp.reshape(window_mapped,(len(it_coord),Bsize))

        #cone extraction for the current batch
        Acones,bcones=Z.get_coneJ((window_mapped),sizeData=time_windows[batch].shape[0])


        theta = Z.thetatilder
        VarZtrx = Z.truncated_covs_between_all_members_of_cone()[1][0][0]
        scorf=tf_unbounded(Acones,bcones,it_params,dim,([inp,*arch]),mask,theta,VarZtrx,m,delta)

        #computes the second term of the obj function (containing the divergence terms)
        kl_mapped = lambda beta: target_func_unbounded_KL(piParams,beta,dim,m) # TODO NNsize scheint nicht benötigt, idk ob dim == NNsize

        #stochastic gradient estimator for the gradient of the expected empirical risk
        val_jest,jest = sge_pwj(scorf,it_params,my_multi_normal,rng)
              
        #computes the value and the gradient (using jax.grad) of the second term of the objective function
        val_grad=kl_mapped(it_params)
        grad= jax.grad(kl_mapped)(it_params)
        #jax.debug.print("grad {}",grad)
      
        #updates the parameter via Adam update rule      
        updates = jest+grad
        updates, opt_state = optimizer.update(updates,opt_state,it_params)  
        it_params = optax.apply_updates(it_params,updates)
        
        return [it_params,opt_state,Acones,bcones,val_jest,val_grad]

    
    min_it_E=0 #will store the epoch index containing the best performing parameters
    min_it_sup=0 #will store the epoch index containing the best performing parameters
    min_error_E=jnp.inf
    min_error_sup=jnp.inf
    milestone_seeds=[]

    #initialization of data structures containing optimization history
    val_jest_epoch= jnp.empty((Z.Nbatches,x_size-2*Z.c_*Z.p))
    val_grad_epoch= jnp.empty((Z.Nbatches,x_size-2*Z.c_*Z.p))

    
    for epoch in trange(1,epochs+1, desc='epochs', colour='green'):

        #creates a new group in the h5 file for the current epoch
        file_epoch= file_m.create_group('epoch'+str(epoch))
      
        val_jest_batch= jnp.empty((0,x_size-2*Z.c_*Z.p))
        val_grad_batch= jnp.empty((0,x_size-2*Z.c_*Z.p))
      
        for batch in range(Z.Nbatches):

            #random PRNG key initialization, ensures reproducibility 
            key = jax.random.PRNGKey(batch+Z.Nbatches*(epoch-1))
            #key mapping, to ensure safe parallelization
            keys = [jax.random.split(key*(el+1), shard_size).reshape(shard_size, 2) for el in range((x_size-2*p*c)//shard_size)]
            
            #batch slicing
            time_mapped=jax.vmap(t_slicing)(time_windows[batch])
            time_mapped=jnp.squeeze(time_mapped,axis=1)     
            
            shard_slicing = lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)
            d_slicing = lambda beta: lax.dynamic_index_in_dim(time_mapped,beta,axis=1)  #  [,:]

            #lambda function, parallelizes the optimization routine to each coordinates shard
            opt_mapping = jax.vmap(lambda og,pmap,opt,rngM: coordit_unbounded(og,pmap,opt,batch,Z.Ncones*Z.a,Z.Ncones,dim,inp,delta,rngM),in_axes=(0,0,0,0))

            output = jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(Ndraws)))
            
            val_jest_stacked= jnp.array([])
            val_grad_stacked= jnp.array([])
            params_stacked =jnp.transpose(jax.vmap(lambda dummy:jnp.array([[],[]]))(range(dim)))

            for ls_i in range((x_size-2*p*c)//shard_size):

              #parallelization is performed serially for each shard, to prevent memory fill
              
              [params[ls_i],opt_state_grid[ls_i],Acones_s[ls_i],bcones_s[ls_i],val_jest,val_grad]= opt_mapping(list_shards[ls_i], params[ls_i], opt_state_grid[ls_i], keys[ls_i])#jax.jit
          
        #mockup of tape_grad    
              params_stacked=jnp.vstack([params_stacked,params[ls_i]])
              val_jest_stacked=jnp.hstack([val_jest_stacked,val_jest])
              val_grad_stacked=jnp.hstack([val_grad_stacked,val_grad])

            val_jest_batch= jnp.vstack([val_jest_batch,val_jest_stacked])
            val_grad_batch= jnp.vstack([val_grad_batch,val_grad_stacked])
      
        val_jest_epoch=jnp.vstack([val_jest_epoch,val_jest_batch])
        val_grad_epoch=jnp.vstack([val_grad_epoch,val_grad_batch])
        

      
        # assuming each epoch is a milestone epoch, we store the value of the objective function, as well as the pac bound 
        milestone_train_error = jnp.array([])
        b_c_e_stacked=jnp.empty((0,Ncones_test-1))
        test_error = jnp.array([])
        bound_train_E = jnp.array([])
        bound_train_sup = jnp.array([])
        bound_test_E = jnp.array([])
        bound_test_sup = jnp.array([])
      
        milestone_seeds.append(jax.random.key(epoch))

        test_slicing = lambda beta: lax.dynamic_index_in_dim(data_test,beta,axis=0)
        
        #lambda function extracting test cones from the test set
        finalStep=lambda coord,beta: ef_setup(coord, beta,Z.a,inp, arch,mask,Ndraws,Ncones_test-1,milestone_seeds[-1])

        
        #lambda function, computes the in-sample linearised pac bound (for the training set)
        e_mapped_train= jax.vmap(lambda  it_pars,it_A,it_b: error_unbounded(it_pars,Z,it_A,it_b,dim,[inp,*arch],mask,piParams,Ncones,delta,rng)) #idk about Acones TODO
            
        #lambda function, computes the out-of-sample linearised pac bound (for the test set)
        e_mapped_test = jax.vmap(lambda  it_pars,it_A,it_b: error_unbounded(it_pars,Z,it_A,it_b,dim,[inp,*arch],mask,piParams,Ncones_test,delta,rng))

        for ls_i in range((x_size-2*p*c)//shard_size): 
          #stores the pac bound values accordingly
              mile_train_e,pac_train_E,pac_train_sup = e_mapped_train(params[ls_i],Acones_s[ls_i],bcones_s[ls_i])
              milestone_train_error=jnp.hstack([milestone_train_error,mile_train_e])
              bound_train_E = jnp.hstack([bound_train_E,pac_train_E])
              bound_train_sup = jnp.hstack([bound_train_sup,pac_train_sup])

          #extraction of the test set cones
              A_c_e,b_c_e=jax.vmap(finalStep)(list_shards[ls_i],params[ls_i]) #save b_c_e_stacked
              b_c_e_stacked=jnp.vstack([b_c_e_stacked,b_c_e])
          #computation of the out-of-sample pac bound
              test_e,pac_test_E, pac_test_sup= e_mapped_test(params[ls_i],A_c_e,b_c_e)
              test_error=jnp.hstack([test_error,test_e])   
              bound_test_E = jnp.hstack([bound_test_E,pac_test_E])  
              bound_test_sup = jnp.hstack([bound_test_sup,pac_test_sup])  

        if jnp.mean(bound_train_E+milestone_train_error,axis=0)<min_error_E: 
        #updates the best parameters                
          min_it_E=epoch
          min_error_E=jnp.mean(bound_train_E+milestone_train_error,axis=0)
          best_params_E=params_stacked

        if jnp.mean(bound_train_sup+milestone_train_error,axis=0)<min_error_sup: 
        #updates the best parameters                
          min_it_sup=epoch
          min_error_sup=jnp.mean(bound_train_E+milestone_train_error,axis=0)
          best_params_sup=params_stacked

      #stores in the current epoch group all the measures for future diagnostic 
        hdata=[Z.a,Z.lambda_, Z.Ncones,bound_train_E,bound_train_sup,milestone_train_error,test_error,bound_test_E,bound_test_sup,val_jest_batch,val_grad_batch,b_c_e_stacked]
        names=[ 'a','lambda', 'm', 'bound_train_E','bound_train_sup', 'train_errors','test_errors','bound_test_E','bound_test_sup','val_jest','val_grad','cones_test']
        jax.debug.print('len hdata {}',len(hdata))
        jax.debug.print('len names {}',len(names))
        makeh5(file_epoch,hdata,names)

    file_last=file_m.create_dataset('last params',data=params_stacked)

    #creates the h5 group for the best parameters
    file_best=file_m.create_group('min')
    file_best.create_dataset('min_error_E',data=min_error_E)
    file_best.create_dataset('min_error_sup',data=min_error_sup)
    file_best.create_dataset('min iteration_E',data=min_it_E)
    file_best.create_dataset('min iteration_sup',data=min_it_sup)
    file_best.create_dataset('best params_E',data=best_params_E)
    file_best.create_dataset('best params_sup',data=best_params_sup)
        
