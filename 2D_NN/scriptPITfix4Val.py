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
datasetsM=['NIGdiamonddata1A4mln','Gaudiamonddata1A4mln']
#datasetsM=['Trappo']
A_estimatedM=[3.840956,3.868912]	
c_estimatedM=[1,1]#

redux=100000
Ndraws=50
Ncones= 500 #how many cones embedded for each batch
mincones=50
maxcones=500
stepcones=50

mina=5
maxa=105
stepa=10

Ncoords=201
lr=0.01
h_t=0.05
eps=5
#a=[47,45,47,47,134,207]#47#3
a=[3,3] #[3,3]
center_pixel=100
p=[1] #try different values

def get_simulated_data(filename):
  data=loadmat(filename+'.mat')
  data=data["data"]
  return data

def rescalingU(d,eps=0):
  m=np.amin(d)
  M=np.amax(d)
  p=(1-2*eps)/(M-m)
  q=(M*eps-(1-eps)*m)/(M-m)
  return d*p+q,p,q

def rescalingInv(d,slope,q,eps=0):
  
  #print(d,slope,q)
  #m=np.amin(d)
  #M=np.amax(d)
  #p=(1-2*eps)/(M-m)
  #q=(M*eps-(1-eps)*m)/(M-m)
  return (d - q)/slope
	



def dim(net):
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
inp=3 #a(p,c), change this if you change p and c
##m=range(mincones,maxcones+stepcones,stepcones)
#a= range(mina,maxa,stepa)
output=[]
bound=[]
archs=[[1]]#,[10,1],[10,10,1],[11,10,10,10,10,10,1],[60,50,30,20,10,1]] #[10,10,1],
#shard_size_p2=[196,196,98,49,28]
shard_size=[198,198,99,66,33]# parallelization, has to be divisors of 198
m=30#[5,10,25,50]#[50,70,85,100]#
piRescaling=range(10,int(np.floor(3.5*m)),10) #gridsearch for prior

file_=h5py.File(path+filename+'tvalidationFix75a3p2.h5','w')   

for i,arch in enumerate(reversed(archs)):#reversed
  print(arch)
  file_net=file_.create_group(str(dim([inp,*arch])))
  for current_id in reversed(range(len(datasetsM))):
    print(datasetsM[current_id])
    data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
    data=data[:,-redux:] #not using first part of the dataset
    data,slope,q=rescalingU(data,eps=0.000001) #rescales values of the field
    last_cone=data[:,-a[current_id]:] 
    data=data[:,:-a[current_id]]
    #print(slope,q)
    #print(data.shape)
    test=last_cone[:,-1] #output value of the last cone (regarding the complete dataset)
    val=data[:,-1] #output value of the second last cone (regarding the complete dataset) #it's not split from data, gets split of in OptSGD, here we only create this variable to store in file
    #print(test)
    N=data.shape[1]
    x_size=data.shape[0]
    c=1
    file_data=file_net.create_group(str(datasetsM[current_id]))
    #print(m)
    for pir in piRescaling:
      print(1./pir)
      file_pir=file_data.create_group('pir'+str(pir))
      #print(file_pir.keys())
      
      #file_pir=file_data.create_group('pir'+str(pir))
      #print(file_pir.keys())
      Z = STOU(0,data,A_estimatedM[current_id],c_estimatedM[current_id],arch,N,m,a[current_id],p[0],h_t=0.05)
      out,params,b,pit,batch,Xt= OptSGD(Z,x_size,'_',eps,delta,data,inp,p[0],c,arch,dim([inp,*arch]),Ndraws,m,Ncoords,shard_size_p2[4-i],lr,rhoScaling,slope,q,piScaling=pir,acrit=acrit) #NCoords not needed, Ndraws number of members in the ensemble forecast(doesn't have something to do with N)
      #outIR=rescalingInv(out,slope,q,eps=0.000001) 
      
      outV,pitV,XtV=testing(Z,last_cone,params,inp,p[0] ,c,arch,dim([inp,*arch]),Ndraws,acrit)
      
      hdata=[out,      outV,  b,      params,  pit,pitV, val,test,  np.array(batch),Xt,XtV,acrit]
      names=['output','outV','bound','params','pit','pitV','test','val','batch',        'pval','pvalV','acrit']
      makeh5(file_pir,hdata,names,path) # SET BACK!!!

#print('no kl annealing')
file_.close()

