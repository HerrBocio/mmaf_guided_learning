import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'#,2,3'#,2,3'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

from STOUpozo import STOU
from SGEbatchPIT import OptSGD,testing
import numpy as np
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import chisquare#kstest as kstest
from scipy.stats import randint
from scipy.stats import kstest
data_path="datasets/"
path='Results/piRs/'
normalize_data=False
use_different_eps=False

#datasets=["Gaudiamonddata1A4","NIGdiamonddata1A4L","Gaudiamonddata10","NIGdiamonddata10","NIGdiamonddataBis1","NIGdiamonddata1Long"]#]
datasetsM=['NIGdiamonddata1A4mln','Gaudiamonddata1A4mln'] #j the datasets we both predict on
#datasetsM=['Trappo']
A_estimatedM=[3.840956,3.868912]	
c_estimatedM=[1,1] #j actually, the estimated c's would be <1, but then the cones would be empty, so we work with c=1, which are the real values of c (with which the datasets had been created)

redux=100000 #j we will cut out all the data from before that timestamp
Ndraws=50 #j number of members of the ensemble forecasts
Ncones= 500 #how many cones embedded for each batch #j batch = minibatch
#j the following 3 lines: not relevant in this version of the script. gridsearch for different numbers of cones in each batch (?)
mincones=50 
maxcones=500
stepcones=50

#j the following 3 lines: not relevant in this version of the script. gridsearch for different values of a (?)
mina=5
maxa=105
stepa=10

Ncoords=201 #j the number of pixels on our lattice
lr=0.01 #j learning rate
h_t=0.05
eps=5
#a=[47,45,47,47,134,207]#47#3
a=[3,3] #j here, a is chosen equal to 3 for both datasets. normally, one would choose a like in Remark 3.9
center_pixel=100 #j not relevant (?)
p=[1] #j one could try different values for p. if one does, inp has to be changed!!

def get_simulated_data(filename):
  data=loadmat(filename+'.mat')
  data=data["data"]
  return data

def rescalingU(d,eps=0):
  """#j
  the procedure works best if the values of the dataset are rescaled in such a way that they all lie between 0 and 1 (??)

  d: data
  eps: ???
  """
  m=np.amin(d)
  M=np.amax(d)
  p=(1-2*eps)/(M-m)
  q=(M*eps-(1-eps)*m)/(M-m)
  return d*p+q,p,q

def rescalingInv(d,slope,q,eps=0):
  """"#j
  reverses rescalingU

  d: data
  slope: ???
  q: ???
  eps: ???
  """
  
  #print(d,slope,q)
  #m=np.amin(d)
  #M=np.amax(d)
  #p=(1-2*eps)/(M-m)
  #q=(M*eps-(1-eps)*m)/(M-m)
  return (d - q)/slope
	



def dim(net):
  '''
  returns number of parameters in the network
  '''
  d=0
  for i in range(len(net)-1):
    d+=(net[i]+1)*net[i+1]
  return d

def makeh5(net,hdata,names,path):
	
  #for i in range(len(datasets)):
  #x=net.create_group(str(dataset))
  for j in range(len(names)):
    net.create_dataset(names[j],data=hdata[j])
  #file_.close()


filename=''


delta=2/.05
#rhoScaling=1./2			
acrit=.75
inp=3 #j equals a(p,c) from the paper. has to be changed if p or c change #i write a function that computes inp
##m=range(mincones,maxcones+stepcones,stepcones)
#a= range(mina,maxa,stepa)
output=[]
bound=[]
#j archs is a list of different NN architectures. arch = [a_1,...,a_n] is a NN with n-1 hidden layers with the ith layer having a_i nodes. a_n is the output layer and always has to be equal to 1
archs = [[1],[10,1],[10,10,1],[11,10,10,10,10,10,1],[60,50,30,20,10,1]] #j different NN architectures
archs= [[1]] #j only working with linear predictors
#shard_size_p2=[196,196,98,49,28]
shard_size=[198,198,99,66,33]#j for parallelization wrt the pixels. normally, we would have 199 pixels inside the frame, but as 199 is prime and the shard sizes (?) have to be divisors of this number, we work with 198
m=30#[5,10,25,50]#[50,70,85,100]#
piRescaling=range(10,int(np.floor(3.5*m)),10) #j gridsearch for prior. the mean of the prior is always zero, these are different values for the variance

file_=h5py.File(path+filename+'tvalidationFix75a3p2.h5','w')   

for i,arch in enumerate(reversed(archs)):#reversed
  print(arch)
  file_net=file_.create_group(str(dim([inp,*arch])))
  for current_id in reversed(range(len(datasetsM))):
    print(datasetsM[current_id])
    data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
    data=data[:,-redux:] #j cutting out the first part of the dataset to make it shorter
    data,slope,q=rescalingU(data,eps=0.000001) #j rescales values of the data such that they are between ... (?)
    last_cone=data[:,-a[current_id]:] #j this will be the test set
    data=data[:,:-a[current_id]] #j removes the last cone from data
    #print(slope,q)
    #print(data.shape)
    Ytest=last_cone[:,-1] #j output value of the test cone
    val=data[:,-1] #j the second last cone of the complete dataset (before line 132) will be the validation set. this is its output value. Notice it is not split from data here, it will actually get split off in OptSGD, here we only create this variable to store its value in file
    #print(test)
    N=data.shape[1] #j like in the paper, the number of timestamps in our dataset
    x_size=data.shape[0] #j the cardinality of |L
    c=1 #i why don't we use c_estimatedM?
    file_data=file_net.create_group(str(datasetsM[current_id]))
    #print(m)
    for pir in piRescaling:
      print(1./pir)
      file_pir=file_data.create_group('pir'+str(pir))
      #print(file_pir.keys())
      
      #file_pir=file_data.create_group('pir'+str(pir))
      #print(file_pir.keys())
      Z = STOU(0,data,A_estimatedM[current_id],c_estimatedM[current_id],arch,N,m,a[current_id],p[0],h_t=0.05)
      out,params,b,pit,batch,Xt= OptSGD(Z,x_size,'_',eps,delta,data,inp,p[0],c,arch,dim([inp,*arch]),Ndraws,m,Ncoords,shard_size_p2[4-i],lr,rhoScaling,slope,q,piScaling=pir,acrit=acrit) #i NCoords not needed
      #outIR=rescalingInv(out,slope,q,eps=0.000001) 
      
      outV,pitV,XtV=testing(Z,last_cone,params,inp,p[0] ,c,arch,dim([inp,*arch]),Ndraws,acrit)
      
      hdata=[out,      outV,  b,      params,  pit,pitV, val,Ytest,  np.array(batch),Xt,XtV,acrit]
      names=['output','outV','bound','params','pit','pitV','test','val','batch',        'pval','pvalV','acrit']
      makeh5(file_pir,hdata,names,path) # SET BACK!!!

#print('no kl annealing')
file_.close()

