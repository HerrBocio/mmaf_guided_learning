import os
import numpy as np
#from scipy.io import loadmat
#from tqdm import trange,tqdm
import jax.numpy as jnp
import jax
from jax import lax
from jax import jit
#import tensorflow as tf
#import scipy#import scipy.linalg.pinv
#from sklearn.linear_model import LinearRegression

#from samplerReview import get_gaussian_sampler, Sampler
import time

#from samplerFFNNg import get_gaussian_sampler_ffnn, Sampler#,get_posterior_ffnn,

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'


@jit
def ffnnPozzo(inp,w):#,bias):
	
    x=inp#arches=arch
    #print(w)
    #print('in pozzo',inp)
    for el in w[:-1]:
       #print('el',el)
       #print(el[:-1,:])
       #print(el.shape)
       
       x=jnp.matmul(x,el[:-1,:])+el[-1,:]
       #print(x.shape)
       activation= lambda vec: (vec<0)*0+(vec>=0)*vec 
       #print('x',x)
       x=jax.vmap(jnp.tanh)(x)  #activation
       #print('ax',x.shape)
    #print('s',jnp.matmul(x,w[-1][:-1,:]))
    return jnp.matmul(x,w[-1][:-1,:]) #q no bias in last layer??
    


@jit
def ffnnLossPozzo(inp,weights,b,eps):	
    #print('inp',inp)#print(arch)
    forward=lambda alpha:(ffnnPozzo(alpha,weights)) #JITTED jax.jit
    l=0
    fun_b = lambda x: (x <= eps).astype(dtype='float32') * x + (x > eps).astype(dtype='float32') * eps
    fun_abs= lambda abs: jnp.abs(abs)
    l=jax.vmap(fun_abs)(jax.vmap(forward,in_axes=1)(inp)-b)
    l=jax.vmap(fun_b)(l)
    l=jnp.mean(l)#-b
    #l=jnp.abs(l)
    return l


def ffnnV(inp,arch,mask,weights):
    #print(weights.shape)
      
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo

    #print(mask,type(mask))
    #print(arch)
    parPozzo=lambda alpha: pozzo(alpha,mask,arch)
    weights=jax.vmap(parPozzo)(weights)

    #print(weights)
  
    forward=lambda alpha: ffnnPozzo(inp,alpha)
    return jnp.transpose(jax.vmap(forward)(weights))


  
def ffnnLoss(inp,arch,weights,b):
    forward=lambda alpha: (ffnn2(alpha,arch,weights)) #
    l=0
    l=jax.vmap(forward)(inp)
    l=jnp.mean(l-b)
    return l






def get_gaussian_sampler_ffnn(dim, mean=0.0,stdev=1.):
    """
    Returns a generator for simulating dim-dimensional normally distributed random vectors
    :param dim:
    :param mean:
    :return:
    """
    #dims= lambda batch_size: 
    print(dim)
    fun = lambda batch_size: jax.random.normal(jax.random.key(0), shape=(batch_size,dim)) * stdev + mean
    
    return fun

#@jit(static_argnames=["c","N"])
def sample_fun(realization,g,c,N): # definitely to be replaced ( in favour of some mcmc)
        print("in sample")
        #N_ = max(1,jnp.minimum((N * c * 0.5).astype('int32'), 1000000))
        #N_ = ((N * c * 0.5)>= 10000)*10000 + ((N * c * 0.5)<10000)*(N * c * 0.5).astype('int16')
        #N_= (N_>1)*N_ + (N_ <= 1)*1
        out=[]
        i=0
        current_N = 0
        rng=jax.random.key(0)
        def sample_step(rng):
                print("int sample step","n_c=",c)
                X = realization(1)
                U = jax.random.uniform(rng,1)
                #print('sample : ',self.g(X)/self.n_c,'u',U)
                pos =  U <= g(X) / c*c #(U <= g(X) / n_c)*1+(U > g(X) / n_c)*0
                print('pos',pos)
                
                def true_fun(val):
                  out.append(val)
                  i+=1
                  return
                def false_fun(val):
                    pass
                lax.cond(pos,true_fun,false_fun,X)
                return         
                       
  
        def sample_loop(rng):
            rng,subkey=jax.random.split(rng)
            sample_step(rng)
            
            return rng
        
        def cond_loop(rng):
            return i<N
      
        
        lax.while_loop(cond_loop, sample_loop,rng)
  
        '''
        while current_N < N: # while loop doesn't favour jitting
            rng,subkey=jax.random.split(rng)
            realization,pos,ind = sample_step(rng,prior,g,c,N)
            #print(realization.shape)
            out.append(realization)
            outI.append(pos)
            current_N += ind
            print("post_n: ",current_N)
        '''
        #out = out[:N]
        #print("out of sample")
        return out
    # @tf.function
'''
def sample_step(rng):
        print("int sample step","n_c=",c)
        X = prior(1)
        U = jax.random.uniform(rng,1)
        #print('sample : ',self.g(X)/self.n_c,'u',U)
        pos =  U <= g(X) / c*c #(U <= g(X) / n_c)*1+(U > g(X) / n_c)*0
        print('pos',pos)
        
        def true_fun(val):
          out.append(val)
          i+=1
          return
        def false_fun(val,i):
            pass
        
        return 
'''  

def get_posterior_ffnn(A,b,arch,dim,mask,loss,m,eps, num_realizations,mean,var,Nsample):
    
    r_eps=get_loss_function(A,b,arch,loss,eps=eps)
    prior=get_gaussian_sampler_ffnn(dim)#=mean,var=var)  
    PR= prior(num_realizations)
    def pozzo(params,mask,arch):
     #print(params.shape)
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo
    pozzo4sample=lambda alpha: pozzo(prior(alpha),mask,arch)
    parPozzo= lambda alpha: print(alpha)#pozzo(alpha,mask,arch)
    print('r shape',PR.shape)
    real= jax.vmap(parPozzo)(PR)
    #jax.debug.print('real',real)
    g_unormalized=lambda beta : jnp.exp(-jnp.sqrt(m)*r_eps(beta))
    c=jnp.mean(jax.vmap(g_unormalized)(real))
    #g_normalized= g_unormalized/c
    return sample_fun(pozzo4sample,g_unormalized,c,Nsample)
    




def get_loss_functionGibbs(A,b,arch,loss,eps=2.99): #too nested, might embed into get_posterior_ffnn
    #print(A.shape)
    #fun_sq = lambda beta: ffnnLoss(A,arch,beta,b) 
    fun_map = lambda beta: ffnnLossPozzo(A,beta,b,eps) 

    fun_b = lambda x:  (x <= eps).astype(dtype='float32') * x + (x>eps).astype(dtype='float32')*eps
    r_eps= lambda weights : jax.vmap(fun_b)(lax.map(fun_map,weights))
    return r_eps



class STOU:
    def __init__(self,x_position,data,A_estimated,c_estimated,archs,N,Ncones=1,a=1,p=1,num_realizations=20,h_t=0.05): #,model
        """
        Initializes STOU model
        :param x_position: coordinate to be investigated
        :param data: spatial-temporal data
        :param num_realizations:
        :param max_variogram_value: maximum value for which the variogram will be determined
        """
        #self.estimate_parameters(data,max_variogram_value=max_variogram_value)
        #data=jnp.array(data)
        #print('have a break in it...')
        #time.sleep(5)  
        self.x_position=np.array([x_position])

        self.data=data

        self.num_realizations=num_realizations
        self.h_t=h_t
          
        self.arch=archs
        self.A_=A_estimated
        self.c_=c_estimated
        self.VarLevySeed_ = .5
        self.alpha= jnp.sqrt(self.c_)*self.VarLevySeed_/self.A_
			  #Z.A_=A_estimated[current_id]
			  #Z.c_=c_estimated[current_id]
        self.lambda_ = self.A_ * np.minimum(2.0, self.c_) / 2*self.c_
			  #print(datasets[current_id],'\n')#,Z.lambda_)
        self.N=N
        self.a=a
        self.p=p
        self.Ncones=Ncones
        Bsize=int(self.a*self.Ncones)
        #print(Bsize)
        self.Bsize=Bsize
        self.lastBatch=self.N % self.Bsize
        self.Nbatches=jnp.floor(self.N/self.Bsize).astype('int16')
        
        
    def infer_beta(self,num=10000): #
        #print(self.sampler.n_c)
        beta=self.sampler.sample(num,self.sampler.n_c)
        #beta=tf.reduce_mean(tf.expand_dims(self.sampler.g(beta),1)*beta,0)
        
        print("\t\tshape> ",beta.shape)
        
        return beta
  
    def set_model_for_p_value(self,data,arch,dim,mask,loss,p=1,num_realizations=2,eps=2.99,mean=0.0,var=1,num=100):
        
        self.p=p

        A,b=self.get_coneJ(data,self.N)
        print(A.shape,b.shape)
        #self.A=A
        #self.b=b
        self.m=A.shape[0]
        #print("m: ",self.m,"points considered: ",data.shape[1]/self.m)
        #self.l=tf.cast(tf.floor(self.m/self.k),dtype=tf.float16)
        self.r_eps=get_loss_function(A,b,arch,loss,eps=eps) #j r_eps = r^{\epsilon}(h) empirical error
        #print("out of get_loss")
        return get_posterior_ffnn(A, b,arch,dim,mask,loss,self.m,eps, num_realizations,mean,var,num)#,model
        
    
    def get_cone_shiftJ_3d(self):
        
        coord=jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(3)))
        for t in reversed(range(p)):  # self.p = p
        # aggiunge alla fine di coord il vettore dentro tf.constant()
            coord1=jnp.array([jnp.array([- (t+1), v, u]) for v in range(-int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1), int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1) + 1)  for u in range(-int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1), int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1) + 1)])
            coord=jnp.vstack([coord,coord1])
        self.cone_shift=coord
          
    def get_cone_shiftJ(self):
        coord=[]
        for t in reversed(range(self.p)):
            #print("t: ",t,"pos: ",self.x_position)
            coord.append(jnp.array([[v, - (t + 1)] for v in range(int(-jnp.floor(self.c_*self.p).astype('int16')*(t + 1)), int(jnp.floor(self.c_*self.p).astype('int16')*(t + 1)) + 1)]))
        coord=jnp.concat(coord,axis=0)
        coord=jnp.expand_dims(jnp.expand_dims(coord,0),0)
        self.cone_shift=coord
        #print("coord",coord)

    def get_coneJ_3d(data):

        """
        Extract cones at a given coordinate as a system of linear equations
        :param x:
        :return:
        """
        size=cone_mapped.shape[0]
        c_=c
        cone_ends = jnp.arange((size-1)%a + ((size-1)%a < p)*a , size, a)                     #size % a +a-p-1
        #print("cone ends shape: ",cone_ends)

        cone_ends_shape = jnp.ceil( (size-1) / a)
        #print("cone ends shape: ",cone_ends_shape)
              
        cone_ends_shape = cone_ends_shape.astype('int16')


        #self.rest=(size-self.rest) % a
        #print("cone ends shape: ",cone_ends_shape)
        #print('x',x)
        #y=jnp.expand_dims(x,1)
                
        x=jnp.array([p,p])
        #print(x)
        cone_ends=jnp.expand_dims(cone_ends,1)
        #print('1',cone_ends)
        cone_ends_coordinates=jax.vmap(lambda t : jnp.hstack([t,x]))(cone_ends)
        #print('new', cone_ends_coordinates)
        coord=get_cone_shiftJ_3d()
        #cone_ends_coordinates=jnp.expand_dims(cone_ends_coordinates,2)

        #print(cone_ends_coordinates)
        cone_coordinates=jax.vmap(lambda s: s+coord)(cone_ends_coordinates)
        #print('new cones',cone_coordinates)
        #print(jnp.transpose(data).shape)
        #print('cone ends',cone_ends_coordinates)
        def gather_nd(par,indices):
          #print(indices)
          return(jax.vmap(lambda x: par[x[0],x[1],x[2]])(indices.astype('int16')))
            
        b=gather_nd(data,cone_ends_coordinates)
        
        #b = jnp.take(self.data[x],cone_ends)#,axis=0)
        print("b shape: ",b)
        #print(self.data.shape)
        
        #print('cone',cone_coordinates)
        
        
        A=jax.vmap(lambda c: gather_nd(data,c))(cone_coordinates)
        
        print('A shape', A) 
        return A,b#jnp.squeeze(A,axis=2),jnp.squeeze(b,axis=1) 
        
  

    def get_coneJ(self,data,sizeData):

        """
        Extract cones at a given coordinate as a system of linear equations
        :param x:
        :return:
        """
        #if dim==0: dim=self.N
        #print('making cones...')
        #time.sleep(20)
        #print('dim',dim.shape)
        size=self.Bsize
        #print('bsize',self.Bsize,sizeData)
        if  not sizeData==self.Bsize:
          #print('in last batch')		
          size=self.lastBatch
        
        #print('size',size)
        #print(data.shape)
        cone_ends = jnp.arange(size % self.a + self.p, size, self.a)
        #print("cone ends shape: ",cone_ends)

        cone_ends_shape = (size - 1 - size % self.a) / (self.a)
        #print("cone ends shape: ",cone_ends_shape)
              
        cone_ends_shape = (jnp.floor(cone_ends_shape) + 1).astype('int16')


        #self.rest=(size-self.rest) % a
        #print("cone ends shape: ",cone_ends_shape)
        #print('x',x)
        #y=jnp.expand_dims(x,1)
        
        x=jnp.expand_dims(jnp.expand_dims(jnp.array([jnp.floor(self.c_*self.p).astype('int16')]),1),1)
        
        cone_ends=jnp.expand_dims(jnp.expand_dims(cone_ends,1),0)
        #print(cone_ends.shape)
        
        x=jnp.broadcast_to(x,[x.shape[0],cone_ends_shape,1])
        #print(x.shape)
        cone_ends=jnp.broadcast_to(cone_ends,x.shape)
        #print(cone_ends)
        cone_ends_coordinates=jnp.concat( (x, cone_ends),axis=2)
        #print(cone_ends_coordinates.reshape((cone_ends_coordinates.shape[1],cone_ends_coordinates.shape[2])).shape)#,cone_ends_coordinates)
        
      
        #print(jnp.transpose(data).shape)
        #print('cone ends',cone_ends_coordinates)
        def gather_nd(params,indices):
            #print(params.shape)
            flat_idx=jnp.ravel_multi_index(indices.T,params.shape)#,order='F')
            return params.flatten()[flat_idx]	

        #print(cone_ends_coordinates)
        b=gather_nd(data,cone_ends_coordinates)
        
        #b = jnp.take(self.data[x],cone_ends)#,axis=0)
        #print("b shape: ",b.shape)
        #print(self.data.shape)
        self.get_cone_shiftJ()
        cone_ends_coordinates=jnp.expand_dims(cone_ends_coordinates,2)
        cone_coordinates=cone_ends_coordinates+self.cone_shift
        #print('cone',cone_coordinates)
        
        
        A=gather_nd(data,cone_coordinates)
        
      
        #print('shape a b',jnp.squeeze(A,axis=2),jnp.squeeze(b,axis=1))
               
        return jnp.squeeze(A,axis=2),jnp.squeeze(b,axis=1) 
        
  

    def estimate_remaining_parameters(self):
        a, k = self.helper()
        self.a = a
        self.k = k



    def estimate_a_k(self,c_id,p,cut,eps):#,flagG,flagN):
        #print("flagG",flagG,"flagN",flagN)
        if c_id>=0:# and flagG==0:
            #print("in est Gau")
            #print("in est Gau",file=outfile)

            a_, k = self.helper_new_Gau(p,eps)
            #flagG=1
            self.a = a_
            self.k = k
            #print("exit est G")
            #print("exit est G",file=outfile)

        if c_id<0:# and flagN==0:
            #print("in est Nig")
            #print("in est Nig",file=outfile)

            a_, k = self.helper_new_Nig(p,eps)
            #flagN=1
            self.a = a_
            self.k = k
            #print("exit est N")
            #print("exit est N",file=outfile)

        #return flagG,flagN



    def helper_new_Nig(self,p,eps=2.99):
         h_t=self.h_t
         p_t = p/h_t
         # val=(self.lambda_*self.p-1)+np.sqrt((self.lambda_*self.p-1)**2.0+4*self.lambda_*h_t*(e+1)/e*self.N)
         
         #print("in Hnig")
         
         val= np.exp(2*eps/self.lambda_)/h_t + p_t
         
         #print("stuck nig?")
         
         k=0
         a_final=0
         found_value=False
         while True:
             k=k+1
             #print("new k: ",k)
             #print("new k: ",k,file=outfile)
             for a_ in range(p+1,min(800,self.N-2+1)):
                 #print("\ta prop: ", a_)
                 #print("\ta prop: ",a_,file=outfile)
                 if a_*k>=val:
                     a_final=a_
                     found_value=True
                     break
             if found_value:
                 break
         # for k in range
    
         return a_final,k

    def helper_new_Gau(self,p,eps=2.99):
         h_t=self.h_t
         p_t=p*h_t
         # val=(self.lambda_*self.p-1)+np.sqrt((self.lambda_*self.p-1)**2.0+4*self.lambda_*h_t*(e+1)/e*self.N)
         #val=(self.lambda_*self.p*h_t-1.0)+ np.sqrt( (self.lambda_*self.p*h_t-1.0)**2.0+8.0*h_t*self.lambda_*self.N )
         
         val= 2*eps/(self.lambda_ * h_t) + p
                 
         k=0
         a_final=0
         found_value=False
         while True:
             k=k+1
             #print("new k: ",k)
             #print("new k: ",k,file=outfile)
             for a_ in range(p+1,min(800,self.N-2+1)):
                 #print("\ta prop: ", a_)
                 #print("\ta prop: ",a_,file=outfile)
                 if a_*k>=val:
                     a_final=a_
                     found_value=True
                     break
             if found_value:
                 break
         # for k in range
         return a_final,k
    
    
    def helper(self,e=33.0):
        h_t=self.h_t
        # val=(self.lambda_*self.p-1)+np.sqrt((self.lambda_*self.p-1)**2.0+4*self.lambda_*h_t*(e+1)/e*self.N)
        val=(self.lambda_*self.p*h_t-1.0)+ np.sqrt( (self.lambda_*self.p*h_t-1.0)**2.0+8.0*h_t*self.lambda_*self.N )
        val=val/(2.0*self.lambda_*h_t)

        k=0
        a_final=0
        found_value=False
        while True:
            k=k+1
            for a in range(self.p+1,self.N-2+1):
                if a*k>=val:
                    a_final=a
                    found_value=True
                    break
            if found_value:
                break
        # for k in range
        return a_final,k  

    def truncated_cov(self, u, tau, r):
        """
        returns Cov(Z_t(x)^(r), Z_{t+tau}(x+u)^(r)) = Var(Lambda') exp(-Au) int_{A_0(0)\V_{(0,0)}^r \cap A_{tau}(u)\V__{(tau,u)}^r} exp(2As) ds
        """
         # the formula below works for tau<=0, u in |R. If tau>0, we have to set tau=-tau, u=-u, as Cov(Z_tau(u)^r, Z_0(0)^r) = Cov(Z_0(0)^r, Z_{-tau}(-u)^r) because of stationarity
        if tau > 0:
            tau = -tau
            u = -u
        #r = a-p
        if tau <= -r:
            return 0
        int = self.c_/self.A_ * (-np.exp(-2*self.A_*r)*(tau+r+1/(2*self.A_)) + np.exp(2*self.A_*tau)/(2*self.A_))

        return self.VarLevySeed_ * np.exp(-self.A_*u) * int
    
    def truncated_covs_between_all_members_of_cone(self, h_s, h_t, r):
        """
        
        """
        distances_XY = []
        for t in reversed(range(self.p)): # t+1 in {p, p-1, p-2, ..., 1}
            b = np.floor(self.c_*(t+1)*h_t/h_s) # b:= argmax {a: a*h_s <= (t+1)*c*h_t}
            distances_XY.append(jnp.array([[v, -h_t*(t+1)] for v in jnp.arange(-b*h_s, (b+1)*h_s, h_s)])) # [spatial pos, temporal pos]
        distances_XY = jnp.concat(distances_XY, axis=0)
        #distances = jnp.expand_dims(jnp.expand_dims(distances, 0), 0) würde zusätzliche dimension hinzufügen
        covs_XY = jnp.array([self.truncated_cov(u=dist[0], tau=dist[1], r=r) for dist in distances_XY])

        distances_XX = []
        for t in reversed(range(self.p)): # t+1 in {p, p-1, p-2, ..., 1}
            b = np.floor(self.c_*(t+1)*h_t/h_s) # b:= argmax {a: a*h_s <= (t+1)*c*h_t}
            distances_XY.append(jnp.array([[v, -h_t*(t+1)] for v in jnp.arange(-b*h_s, (b+1)*h_s, h_s)])) # [spatial pos, temporal pos]
        distances_XY = jnp.concat(distances_XY, axis=0)

        return covs_XY
