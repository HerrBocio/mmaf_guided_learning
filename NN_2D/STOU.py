import os
import numpy as np
 
import jax.numpy as jnp
import jax
from jax import lax
from jax import jit
 
import time
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'


@jit
def ffnn_forward_pass(inp,w):#,bias):
    '''
    Function that computes the forward pass for a generic neural network
    Input:
        inp: input data for the ffnn
        w: network parameters
    '''
    x=inp
    for el in w[:-1]:
       x=jnp.matmul(x,el[:-1,:]) +el[-1,:]
       activation= lambda vec: (vec<0)*0+(vec>=0)*vec 
       x=jax.vmap(activation)(x)  #activation
    return jnp.matmul(x,w[-1][:-1,:])

@jit
def ffnn_loss_forward_pass(inp,weights,b,eps):	
    '''
    Computes montecarlo estimator for the empirical risk of the training data 
    '''
    forward=lambda alpha:(ffnn_forward_pass(alpha,weights))
    fun_b = lambda x: (x <= eps).astype(dtype='float32') * x + (x > eps).astype(dtype='float32') * eps
    l=0
    forward_mapped=jax.vmap(forward,in_axes=1)(inp)
    forward_mapped=forward_mapped.reshape((forward_mapped.shape[0]))
    l=jax.vmap(jnp.abs)(forward_mapped-b)
    l=jax.vmap(fun_b)(l)
    l=jnp.mean(l)#-b
    return l


def ffnnV(inp,arch,mask,weights):
      
    def pozzo(params,mask,arch):
     pozzo=[params[mask[el-1]:mask[el]].reshape((arch[el-1]+1,arch[el])) for el in range(1,len(mask))]
     return pozzo

    parPozzo=lambda alpha: pozzo(alpha,mask,arch)
    weights=jax.vmap(parPozzo)(weights)

    forward=lambda alpha: ffnn_forward_pass(inp,alpha)
    return jnp.transpose(jax.vmap(forward)(weights))


class STOU:
    '''
    Class of all the methods for mmaf guided learning
    '''
    
    def __init__(self,A_estimated,c_estimated,arch,N,N_last_cones=50,Ncones=1,a=1,p=1,h_t=0.05): 
        '''
        Class constructor
        Inputs: 
            A_estimated: estimate of the hyperparameter A
            c_estimated: estimate of the hyperparameter c
            archs: network layers structure
            N: size of the dataset
            N_last_cones: number of test cones
            Ncones: number of training cones for each batch
            a: gap between cones
            p: cone length
            h_t: data discretization step
              
        '''
        self.A_=A_estimated
        self.c_=c_estimated
        self.h_t=h_t
        self.arch=arch
        self.alpha= jnp.sqrt(self.c_)*.5/self.A_
        self.lambda_ = self.A_ * np.minimum(2.0, self.c_) / (2*self.c_)
        self.N=N
        self.a=a
        self.p=p
        self.Ncones=Ncones
        #computes the size of each batch
        Bsize=int(self.a*self.Ncones)
        self.Bsize=Bsize
        #computes the size of the pretraining batch
        self.pre_train_batch=self.Ncones*self.a
        #copmputes the size of the test batch
        self.lastBatch=N_last_cones*self.a 
      
        self.Nbatches=jnp.floor(self.N/self.Bsize).astype('int16')
        
        self.VarLevySeed_ = 0.5
        self.r=self.a - self.p
        self.thetatilder = jnp.sqrt(self.VarLevySeed_*(self.c_*self.r/self.A_ + self.c_/(2*self.A_**2)))*jnp.exp(-self.A_*self.r)
   
    def get_cone_shiftJ_3d(self):
        '''
        Methods that builds the cone embedding for extracting (X_i,Y_i), in the 3d case
        '''
        coord=jnp.transpose(jax.vmap(lambda dummy: jnp.array([]))(range(3)))
        for t in reversed(range(p)):   
        # at each t adds the relative coordinates of the cone coordinates
            coord1=jnp.array([jnp.array([- (t+1), v, u]) for v in range(-int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1), int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1) + 1)  for u in range(-int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1), int(jnp.floor(c*(t + 1)/jnp.sqrt(2))+1) + 1)])
            coord=jnp.vstack([coord,coord1])
        self.cone_shift=coord

    
      
    def get_cone_shiftJ(self):
        '''
        Methods that builds the cone embedding for extracting (X_i,Y_i), in the 2d case
        '''
        coord=[]
        for t in reversed(range(1,self.p+1)):
          # at each t adds the relative coordinates of the cone coordinates
            coord.append(jnp.array([[v, - (t )] for v in range(int(-jnp.ceil(self.c_*t).astype('int16')), int(jnp.ceil(self.c_*t).astype('int16')) + 1)]))  
        coord=jnp.concat(coord,axis=0)
        coord=jnp.expand_dims(jnp.expand_dims(coord,0),0)
        self.cone_shift=coord

    def get_coneJ_3d(self,data,sizeData,preT=False):

        """
        Extract cones at a given coordinate as a system of linear equations, in 3d case
        Input:
            data: dataset where to extract the training data (X_i,Y_i)
            sizeData: lenght of data in the temporal coordinate
        """
        size=cone_mapped.shape[0]
        #discernes the training task to the test evaluation 
        if  not sizeData==self.Bsize:
          if preT:
            size=self.pre_train_batch
          else:
            size=self.lastBatch
            
        #selecting the coordinate of the cone heaps
        cone_ends = jnp.arange((size-1)%a + ((size-1)%a < p)*a , size, a)                    
        cone_ends_shape = jnp.ceil( (size-1) / a)
        cone_ends_shape = cone_ends_shape.astype('int16')

        #selects the coordinate of the truncated cone
        x=jnp.array([p,p])
        cone_ends=jnp.expand_dims(cone_ends,1)
        cone_ends_coordinates=jax.vmap(lambda t : jnp.hstack([t,x]))(cone_ends)
        coord=get_cone_shiftJ_3d()
        cone_coordinates=jax.vmap(lambda s: s+coord)(cone_ends_coordinates)
        
        def gather_nd(par,indices):
          return(jax.vmap(lambda x: par[x[0],x[1],x[2]])(indices.astype('int16')))

        #extracts the cone heaps Y_i
        b=gather_nd(data,cone_ends_coordinates)
        #extracts the cone truncation X_i
        A=jax.vmap(lambda c: gather_nd(data,c))(cone_coordinates)
        
        return A,b 
  

    def get_coneJ(self,data,sizeData,preT=False):

        """
        Extract cones at a given coordinate as a system of linear equations, in 2d case
        Input:
            data: dataset where to extract the training data (X_i,Y_i)
            sizeData: lenght of data in the temporal coordinate
        """
        
        size=self.Bsize
        if  not sizeData==self.Bsize:
          if preT:
            size=self.pre_train_batch
          else:
            size=self.lastBatch

        #selecting the coordinate of the cone heaps
        cone_ends = jnp.arange(size % self.a + self.p, size, self.a)
        cone_ends_shape = (size - 1 - size % self.a) / (self.a)
        cone_ends_shape = (jnp.floor(cone_ends_shape) + 1).astype('int16')
        if not size%self.a: 
          cone_ends=jnp.flip(jnp.arange(size-1,0,-self.a))
          cone_ends_shape = (size - 1 - size % self.a) / (self.a)
          cone_ends_shape = (jnp.floor(cone_ends_shape) + 1).astype('int16')
        x=jnp.expand_dims(jnp.expand_dims(jnp.array([jnp.ceil(self.c_*self.p).astype('int16')]),1),1)
        cone_ends=jnp.expand_dims(jnp.expand_dims(cone_ends,1),0)
        x=jnp.broadcast_to(x,[x.shape[0],cone_ends_shape,1])
        cone_ends=jnp.broadcast_to(cone_ends,x.shape)
        cone_ends_coordinates=jnp.concat( (x, cone_ends),axis=2)
        def gather_nd(params,indices):
            flat_idx=jnp.ravel_multi_index(indices.T,params.shape)#,order='F')
            return params.flatten()[flat_idx]	
        #extracts the cone heaps Y_i
        b=gather_nd(data,cone_ends_coordinates)
        #selects the coordinate of the truncated cone
        self.get_cone_shiftJ()
        cone_ends_coordinates=jnp.expand_dims(cone_ends_coordinates,2)
        cone_coordinates=cone_ends_coordinates+self.cone_shift
        #extracts the cone truncation X_i
        A=gather_nd(data,cone_coordinates)
               
        return jnp.squeeze(A,axis=2),jnp.squeeze(b,axis=1) 
   
    def truncated_cov(self, u, tau):
        """
        returns Cov(Z_t(x)^(r), Z_{t+tau}(x+u)^(r)) = Var(Lambda') exp(-Au) int_{A_0(0)\V_{(0,0)}^r \cap A_{tau}(u)\V__{(tau,u)}^r} exp(2As) ds
        """
         # the formula below works for tau<=0, u in |R. If tau>0, we have to set tau=-tau, u=-u, as Cov(Z_tau(u)^r, Z_0(0)^r) = Cov(Z_0(0)^r, Z_{-tau}(-u)^r) because of stationarity
        if tau > 0:
            tau = -tau
            u = -u
        #r = a-p
        if tau <= -self.r:
            return 0
        int = self.c_/self.A_ * (-np.exp(-2*self.A_*self.r)*(tau+self.r+1/(2*self.A_)) + np.exp(2*self.A_*tau)/(2*self.A_))

        return self.VarLevySeed_ * np.exp(-self.A_*u) * int
    
    def get_apc(self):
        apc = 0
        for t in reversed(range(self.p)): # t+1 in {p, p-1, p-2, ..., 1}
            apc += 2*np.floor(self.c_*(t+1)*self.h_t/self.h_s) + 1
        return apc # needs to be tested

    def truncated_covs_between_all_members_of_cone(self):
        """
        
        """
        distances_XY = []
        for t in reversed(range(self.p)): # t+1 in {p, p-1, p-2, ..., 1}
            bt = np.floor(self.c_*(t+1)*self.h_t/self.h_s) # b:= argmax {a: a*h_s <= (t+1)*c*h_t}
            distances_XY.append(jnp.array([[v, -self.h_t*(t+1)] for v in jnp.arange(-bt*self.h_s, (bt+1)*self.h_s, self.h_s)])) # [spatial pos, temporal pos]
        distances_XY = jnp.concat(distances_XY, axis=0)
        covs_XY = jnp.array([self.truncated_cov(u=dist[0], tau=dist[1]) for dist in distances_XY])

        distances_XX = []
        covs_XX = []
        for t in range(self.p,0,-1): # t in {p, p-1, p-2, ..., 1}
            bt = int(np.floor(self.c_*t*self.h_t/self.h_s)) # bt:= argmax {a: a*h_s <= (t+1)*c*h_t}
            for pixel1 in jnp.arange(-bt*self.h_s,(bt+1)*self.h_s, self.h_s):
                dist_row = []
                cov_row = []
                for s in range(self.p,0,-1):
                    bs = int(np.floor(self.c_*s*self.h_t/self.h_s))
                    for pixel2 in jnp.arange(-bs*self.h_s, (bs+1)*self.h_s, self.h_s):
                        dist_row.append([float(pixel1-pixel2), -self.h_t*(t-s)])
                        cov_row.append(self.truncated_cov(u = float(pixel1-pixel2), tau = -self.h_t*(t-s)))
                distances_XX.append(dist_row)
                covs_XX.append(cov_row)
        covs_XX = jnp.array(covs_XX)
        
        return covs_XY, covs_XX

    def calc_covs(self):
        self.covs = self.truncated_covs_between_all_members_of_cone()
 