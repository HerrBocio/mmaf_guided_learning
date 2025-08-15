import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
#from STOUpozo import STOU
data_path="datasets/"
normalize_data=False
use_different_eps=False
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'#,2,3'
from scipy.stats import kstest as kstest
from scipy.stats import chisquare
from scipy.stats import randint


size=1000

h_t=0.05


center_pixel=100



datasetsM=['NIGdiamonddata1A4mln','Gaudiamonddata1A4mln']


p=[1]

m=30
inp=3
net=[[inp,1],[inp,10,1],[inp,10,10,1],[inp,11,10,10,10,10,10,1],[inp,60,50,30,20,10,1]] #[10,10,1],

piRescaling=range(10,int(np.floor(3.5*m)),10)

path='Results/piRs/validation/'
pathB='Results/piRs/validation/0.75/a98Full/'
#path=

flag=1
def dumb(archs):
  dim=0
  for i in range(len(archs)-1):
    dim = dim + (archs[i]+1)*archs[i+1]
    #print(dim)
  return dim
  

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
  return (d - q)/slope


headerB=['Tr\net',*list(map(dumb,net))]
headerP=headerB
tableB=[list(piRescaling),list(piRescaling)]
tableP=[list(piRescaling),list(piRescaling)]
Nepochs=2221
redux=1000000
a=[98,99]


if not os.path.exists(pathB):
    os.makedirs(pathB)
    print("folder created")



fB= open(pathB+'validationB.txt','w')
fB.write('Bounds\n')
fP = open(pathB+'validationP.txt','w')
fP.write('p values\n')
    
if not os.path.exists(pathB):
    os.makedirs(pathB)
    print("folder created")
hf=h5py.File(path+'tvalidationFix75a98Full'+'.h5','r')	 # a98
print(hf.keys())
for i in net:#for current_id in ids:
	#for current_id in ids:
	
	print(str(dumb(i)))
	n_g=hf.get(str(dumb(i)))
	print(n_g.keys())
	netB=[[],[]]
	netP=[[],[]]
	#hf=h5py.File(pathB+'ReplotTest_PWJ'+str(i)+'.h5','r')	
	#d_g=hf.get(str(datasetsN[current_id])) 		
	for current_id in range(len(datasetsM)):	#for i in net:
		c_g=n_g.get(str(datasetsM[current_id]))
		print(datasetsM[current_id])
		print(c_g.keys())
		
		Nepochs = int(np.floor(redux/(m*a[current_id]))-1)
		for pir in piRescaling:
			pir_g=c_g.get('pir'+str(pir))
			print(pir_g.keys())
			if not os.path.exists(pathB+str(dumb(i))+'/'+str(pir)):
			  os.makedirs(pathB+str(dumb(i))+'/'+str(pir))#data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
			#data,slope,q=rescalingU(data[:,-200000:],eps=0.000001)
			#print(slope,q)
			#test=data[: ,-1]#print(data.shape)    
			#x_size,N=data.shape
			#Z=STOU(0,data,A_estimated[current_id],c_estimated[current_id],i,N)
			
			#max_val=int(data.shape[1]-1)#int(data.shape[1]*.95)#
			#data_val=data[:,max_val:]

			#data=data[:,:max_val]#-2000
			
			output=pir_g.get('output')
			outV=pir_g.get('outV')#output
			output=output[:,:]
			outV=outV[:,:]
			print(output.shape,outV.shape)
			bou=pir_g.get('bound')
			pit=pir_g.get('pit')
			pitV=pir_g.get('pitV')
			#print(np.array(pitV))
			batch=pir_g.get('batch')
			test=pir_g.get('test')
			val=pir_g.get('val')
			pv_val=np.array(pir_g.get('pvalV'))
			pv= np.array(pir_g.get('pval'))
			
			print(chisquare(pitV).pvalue,chisquare(pit).pvalue)
			plt.figure()
			plt.stairs(np.array(pit),fill=True)
			plt.ylim((-.9,15))
			plt.savefig(pathB+str(dumb(i))+'/'+str(pir)+'/pit_'+datasetsM[current_id]+'.png')
			plt.close()
			plt.figure()
			plt.stairs(np.array(pitV),fill=True)
			plt.ylim((-.9,15))
			plt.savefig(pathB+str(dumb(i))+'/'+str(pir)+'/pitV_'+datasetsM[current_id]+'.png')
			plt.close()
			netB[current_id].append(np.format_float_positional(np.round(np.mean(np.array(bou[:])),decimals=4)))
			netP[current_id].append(np.format_float_positional(np.round(np.array(pv_val),decimals=4)))#.pvalue
            
			outV_005=np.quantile(outV,0.005,axis=1)	#np.array(x.get('q025'))
			outV_995=np.quantile(outV,0.995,axis=1)
			outV_025=np.quantile(outV,0.025,axis=1)	#np.array(x.get('q025'))
			outV_975=np.quantile(outV,0.975,axis=1)#print(pv)
			#print(np.mean(bou[:]),np.array(batch))
			out_005=np.quantile(output,0.005,axis=1)	#np.array(x.get('q025'))
			out_995=np.quantile(output,0.995,axis=1)
			out_025=np.quantile(output,0.025,axis=1)	#np.array(x.get('q025'))
			out_975=np.quantile(output,0.975,axis=1)	
			out_25=np.quantile(output,0.25,axis=1)#np.array(x.get('q975'))
			out_75=np.quantile(output,0.75,axis=1)
			out_45=np.quantile(output,0.45,axis=1)
			out_55=np.quantile(output,0.55,axis=1)
			plt.figure()
    
			#plt.plot
			plt.ylim((-1,2))
			plt.fill_between(np.arange(1,len(out_25)+1),outV_025,outV_975,alpha=0.65,color='teal')			
			#plt.fill_between(np.arange(1,len(out_25)+1),out_025,out_975,alpha=0.65,color='crimson')	str(np.round(np.array(pv),decimals=3)		
			plt.plot(np.arange(1,len(out_25)+1),val[1:-2],linewidth=.8,color='black')
			#if(np.array(batch)<Nepochs):
			plt.annotate('p value for test set ' +str(np.round(pv_val,decimals=3)) ,xy=(120,-0.8),fontsize='x-small')
			#else: plt.annotate('always reject in '+str(Nepochs)+' iterations with bound '+str(np.round(np.mean(bou[:]),decimals=3)),xy=(90,-0.8),fontsize='x-small')
			plt.savefig(pathB+str(dumb(i))+'/'+str(pir)+'/V_'+datasetsM[current_id]+'.png')#.pvalue
			plt.close()

			plt.figure()
    
			plt.plot
			plt.ylim((-1,2))
			plt.fill_between(np.arange(1,len(out_25)+1),out_025,out_975,alpha=0.65,color='teal')			
			#plt.fill_between(np.arange(1,len(out_25)+1),out_025,out_975,alpha=0.65,color='crimson')	str(np.round(np.array(pv),decimals=3)		
			plt.plot(np.arange(1,len(out_25)+1),test[1:-2],linewidth=.8,color='black')
			if(np.array(batch)<Nepochs):
			  plt.annotate('no rejection after '+str(np.array(batch))+' iterations with p value ' +str(np.round(pv,decimals=3)) +' and bound '+str(np.round(np.mean(bou[:]),decimals=3)) ,xy=(60,-0.8),fontsize='x-small')
			else: plt.annotate('always reject in '+str(Nepochs)+' iterations with bound '+str(np.round(np.mean(bou[:]),decimals=3)),xy=(90,-0.8),fontsize='x-small')
			plt.savefig(pathB+str(dumb(i))+'/'+str(pir)+'/'+datasetsM[current_id]+'.png')#.pvalue
			plt.close()
	tableB[0]=np.vstack([tableB[0],netB[0]])
	tableB[1]=np.vstack([tableB[1],netB[1]])
	tableP[0]=np.vstack([tableP[0],netP[0]])
	tableP[1]=np.vstack([tableP[1],netP[1]])

print(datasetsM[0],'\n',np.vstack([headerB,np.transpose(tableB[0])]),'\n',datasetsM[1],'\n',np.vstack([headerB,np.transpose(tableB[1])]),file=fB)
print(datasetsM[0],'\n',np.vstack([headerP,np.transpose(tableP[0])]),'\n',datasetsM[1],'\n',np.vstack([headerP,np.transpose(tableP[1])]),file=fP)

hf.close()
#hfB.close()
