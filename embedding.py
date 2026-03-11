import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
from scipy.optimize import newton
from utils import truncated_cov, theta_r


class Embedding(): 

    def __init__(self, config:dict):
                 
        self.p=config.hparams.p
        self.val_start_idx = config.data.val_start_idx
        self.eps=config.hparams.eps
        self.h_t=config.hparams.h_t
        self.m_batch=config.data.m_batch
      
        self.shard_size=config.shard_size
        self.data_path=config.data.path+config.data.name
        self.data_name=config.data.name
        
        self.num_coords=config.data.num_coords
        self.Ncones_test=config.data.Ncones_test
        

    def gather_1d(self,data, indices):
        flat_idx=jnp.ravel_multi_index(indices.T,data.shape)#,order='F')
        return data.flatten()[flat_idx]	

      
    def gather_2d(self,data,indices):
        return(vmap(lambda x: data[x[0],x[1],x[2]])(indices.astype('int16')))
  

    def get_cone_shiftJ(self):
        '''
        Methods that builds the cone embedding for extracting (X_i,Y_i), in the 2d case
        '''
        coord=[]
        for t in reversed(range(1,self.p+1)):
          # at each t adds the relative coordinates of the cone coordinates
            coord.append(jnp.array([[v, - t ] for v in range(int(-jnp.ceil(self.c*t).astype('int16')), int(jnp.ceil(self.c*t).astype('int16')) + 1)]))  
        coord=jnp.concat(coord,axis=0)
        coord=jnp.expand_dims(jnp.expand_dims(coord,0),0)
        self.cone_shift=coord

      
    def get_cone_shiftJ_3d(self):
        '''
        Methods that builds the cone embedding for extracting (X_i,Y_i), in the 3d case
        '''
        coord=jnp.transpose(vmap(lambda dummy: jnp.array([]))(range(3)))
        for t in reversed(range(self.p)):  # self.p = p
        # aggiunge alla fine di coord il vettore dentro tf.constant()
            coord1=jnp.array([jnp.array([- (t+1), v, u]) for v in range(-int(jnp.floor(self.c*(t + 1)/jnp.sqrt(2))+1), int(jnp.floor(self.c*(t + 1)/jnp.sqrt(2))+1) + 1)  for u in range(-int(jnp.floor(self.c*(t + 1)/jnp.sqrt(2))+1), int(jnp.floor(self.c*(t + 1)/jnp.sqrt(2))+1) + 1)])
            coord=jnp.vstack([coord,coord1])
        #print('coord',coord)
        self.cone_shift=coord

      
    def get_coneJ(self, data, size):

        """
        Extract cones at a given coordinate as a system of linear equations, in 2d case
        Input:
            data: dataset where to extract the training data (X_i,Y_i)
        """
        #selecting the coordinate of the cone heaps
        cone_ends = jnp.flip(jnp.arange(size-1,size % self.a + self.p, -self.a)) # MODIFIED W/ SIZE -> SIZE - 1
        cone_ends_shape = cone_ends.shape[0] #(size - 1 - size % self.a) / (self.a)
        #cone_ends_shape = (jnp.floor(cone_ends_shape) + 1).astype('int16')
        if not size%self.a: 
          cone_ends=jnp.flip(jnp.arange(size-1,0,-self.a))
          cone_ends_shape=cone_ends.shape[0]
        x=jnp.expand_dims(jnp.expand_dims(jnp.array([jnp.ceil(self.c*self.p).astype('int16')]),1),1)
        cone_ends=jnp.expand_dims(jnp.expand_dims(cone_ends,1),0)
        x=jnp.broadcast_to(x,[x.shape[0],cone_ends_shape,1])
        cone_ends=jnp.broadcast_to(cone_ends,x.shape)
        cone_ends_coordinates=jnp.concat( (x, cone_ends),axis=2)
        #print(cone_ends_coordinates)
        #extracts the cone heaps Y_i
        b=self.gather_1d(data,cone_ends_coordinates)
        #selects the coordinate of the truncated cone
        self.get_cone_shiftJ()
        cone_ends_coordinates=jnp.expand_dims(cone_ends_coordinates,2)
        cone_coordinates=cone_ends_coordinates+self.cone_shift
        #print(cone_coordinates)
        #extracts the cone truncation X_i
        A=self.gather_1d(data,cone_coordinates)
        emb=jnp.vstack([jnp.squeeze(A,axis=2),jnp.squeeze(b,axis=1)])
        #emb=jnp.hstack([jnp.transpose(jnp.squeeze(A,axis=2)),jnp.expand_dims(jnp.squeeze(b,axis=1),axis=-1)])
        return emb

      
    def get_coneJ_3d(self, data, size):

        """
        Extract cones at a given coordinate as a system of linear equations, in 3d case
        Input:
            data: dataset where to extract the training data (X_i,Y_i)
        """
            
        #selecting the coordinate of the cone heaps
        cone_ends = jnp.arange((size-1)%self.a + ((size-1)%self.a < self.p)*self.a , size, self.a)
        cone_ends_shape = jnp.ceil( (size-1) / self.a)
        cone_ends_shape = cone_ends_shape.astype('int16')
        if not size%self.a: 
            # the coordinates differ in case of validation/test cones
            # since the initial shift is not considered, and the 
            # first cone is at distance a_val
            cone_ends=jnp.flip(jnp.arange(size-1,0,-self.a))
            cone_ends_shape = (size - 1 - size % self.a) / (self.a)
            cone_ends_shape = (jnp.floor(cone_ends_shape) + 1).astype('int16')
  
        #selects the coordinate of the truncated cone
        x=jnp.array([self.p,self.p])
        cone_ends=jnp.expand_dims(cone_ends,1)
        cone_ends_coordinates=vmap(lambda t : jnp.hstack([t,x]))(cone_ends)
        self.get_cone_shiftJ_3d()
        cone_coordinates=vmap(lambda s: s+self.cone_shift)(cone_ends_coordinates)
        
       

        #extracts the cone heaps Y_i
        b=self.gather_2d(data,cone_ends_coordinates)
        #extracts the cone truncation X_i
        A=vmap(lambda c_idx: self.gather_2d(data,c_idx))(cone_coordinates)
        
        return A,b 

  
    def embedded_data(self):

      data=np.load(self.data_path+'/data.npy')
      if self.data_name=='Gaudiamonddata1A4mln':
        self.A=3.840956
        hatc=1
        VarLevySeed = 0.5 # TODO is that correct?
      elif self.data_name== 'NIGdiamonddata1A4mln':
        self.A=3.868912
        hatc=1  
        VarLevySeed = 0.5 # TODO is that correct?
      else:  
        VarLevySeed = 0.5 # TODO what is it??
        Y=np.array(list(data))
        nrY, ncY = Y.shape
        
        dt=2
        dy=10
        
        ndata = nrY * ncY

        
        s1 = np.nansum(Y)
        s2 = np.nansum(Y**2)
        s3 = np.nansum(Y**3)
        s4 = np.nansum(Y**4)
        
        k2 = (1 / (ndata * (ndata - 1))) * (ndata * s2 - s1**2)
        
        d01 = Y.copy()
        
        d01[:, 0:ncY-dt] = d01[:, dt:ncY]
        d01[:, ncY-dt:] = np.nan
        g01 = np.nanmean((Y - d01)**2) / k2
        
        d10 = Y.copy()
        d10[0:nrY-dy, :] = d10[dy:nrY, :]
        d10[nrY-dy:, :] = np.nan
        g10 = np.nanmean((Y - d10)**2) / k2
        
        self.A = -np.log(1 - g01 / 2) / ( dt)
        hatc = -(self.A * dy) / np.log(1 - g10 / 2)

      
      lambda_=self.A * np.minimum(2.0, hatc) / (2*hatc)
      
      self.c=int(np.floor(hatc))
        
      data=data[:self.num_coords+2*self.c*self.p,:]
        
      print(lambda_,self.A,hatc)

      if self.data_name=='Gaudiamonddata1A4mln' or  self.data_name=='NIGdiamonddata1A4mln':
        a_search=lambda x : lambda_*self.h_t*(x-self.p) + np.log(0.025/(2*self.eps*self.m_batch)) 
        self.a = int(np.ceil(newton(a_search,1)))
      elif self.data_name=='OLR_full':
        a_search=lambda x : jnp.round(lambda_,decimals=3)*self.h_t*(x-self.p) + np.log(0.025*x/(2*self.eps*self.val_start_idx)) 
        self.a = int(np.ceil(newton(a_search,1)))
      
      self.Ncones = int(self.val_start_idx//self.a) 
      self.Nbatches=self.Ncones//self.m_batch
      print(self.a,self.Nbatches,self.Ncones,self.data_name)
      list_shards= ([jnp.array([jnp.arange(coord-self.p*self.c,coord+self.p*self.c+1) for coord in range(element, element+self.shard_size)]) for element in range(self.p*self.c, data.shape[0]-self.p*self.c, self.shard_size)])
      clean_data=[]
      for ls in list_shards:
        data_sharded=[]
        for t_batch_idx in range(self.Nbatches):
          data_stacked= jnp.empty((0,2*self.c*self.p+1,self.a*self.m_batch))  #self.val_start_idx-1))
          for i in ls:
            data_stacked=jnp.vstack([data_stacked,data[i, t_batch_idx*self.m_batch*self.a:(t_batch_idx+1)*self.m_batch*self.a].reshape(1, *data[i,t_batch_idx*self.m_batch*self.a:(t_batch_idx+1)*self.m_batch*self.a].shape)])
          data_sharded.append(data_stacked)
        data_mapped = lambda alpha: self.get_coneJ(alpha,self.a*self.m_batch)  #self.val_start_idx-1)
        clean_data_t = [vmap(data_mapped)(el) for el in data_sharded]#
        #print('clean',len(clean_data_t),clean_data_t[0].shape)
        clean_data.append(clean_data_t)
        #print(len(clean_data))
      self.clean_data=clean_data
      
      data_test_sharded=[]
      for ls in list_shards:
        data_stacked= jnp.empty((0,2*self.c*self.p+1,data[:,self.val_start_idx:].shape[-1]))
        for i in ls:
          data_stacked=jnp.vstack([data_stacked,data[i,self.val_start_idx:].reshape(1,*data[i,self.val_start_idx:].shape)])
        data_test_sharded.append(data_stacked)
      data_test_mapped = lambda alpha: self.get_coneJ(alpha,data[:,self.val_start_idx:].shape[-1])
      self.clean_data_test = [vmap(data_test_mapped)(el) for el in data_test_sharded]#
      
      self.theta = theta_r(self.A, hatc, self.a-self.p, VarLevySeed) 
      self.VarZtrx = truncated_cov(self.A, hatc, self.a-self.p, 0, 0, VarLevySeed)




      
