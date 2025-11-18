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
from NN_2D.variables import datasetsM, h_t, center_pixel, p, m, inp, piRescaling, rescalingU, rescalingInv, dim, redux, a, archs

net=[[inp,*arch] for arch in archs]


#path='Results/piRs/validation/'
path='Results/piRs/'
#pathB='Results/piRs/validation/0.75/a98Full/'
pathB=path

flag=1
  

def get_simulated_data(filename):
    data=loadmat(filename+'.mat')
    data=data["data"]

    return data


headerB=['Tr\net',*list(map(dim,net))]
headerP=headerB
tableB=[list(piRescaling),list(piRescaling)]
tableP=[list(piRescaling),list(piRescaling)]
#Nepochs=2221

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
#hf=h5py.File(path+'tvalidationFix75a98Full'+'.h5','r')	 # a98
hf=h5py.File(path+'tvalidationFix75a3p2'+'.h5','r')
print(hf.keys())
for i in net:#for current_id in ids:
	#for current_id in ids:
	
	print(str(dimComp(i)))
	n_g=hf.get(str(dimComp(i)))
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
			if not os.path.exists(pathB+str(dimComp(i))+'/'+str(pir)):
			  os.makedirs(pathB+str(dimComp(i))+'/'+str(pir))#data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
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
			plt.savefig(pathB+str(dimComp(i))+'/'+str(pir)+'/pit_'+datasetsM[current_id]+'.png')
			plt.close()
			plt.figure()
			plt.stairs(np.array(pitV),fill=True)
			plt.ylim((-.9,15))
			plt.savefig(pathB+str(dimComp(i))+'/'+str(pir)+'/pitV_'+datasetsM[current_id]+'.png')
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
			plt.savefig(pathB+str(dimComp(i))+'/'+str(pir)+'/V_'+datasetsM[current_id]+'.png')#.pvalue
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
			plt.savefig(pathB+str(dimComp(i))+'/'+str(pir)+'/'+datasetsM[current_id]+'.png')#.pvalue
			plt.close()
	tableB[0]=np.vstack([tableB[0],netB[0]])
	tableB[1]=np.vstack([tableB[1],netB[1]])
	tableP[0]=np.vstack([tableP[0],netP[0]])
	tableP[1]=np.vstack([tableP[1],netP[1]])

print(datasetsM[0],'\n',np.vstack([headerB,np.transpose(tableB[0])]),'\n',datasetsM[1],'\n',np.vstack([headerB,np.transpose(tableB[1])]),file=fB)
print(datasetsM[0],'\n',np.vstack([headerP,np.transpose(tableP[0])]),'\n',datasetsM[1],'\n',np.vstack([headerP,np.transpose(tableP[1])]),file=fP)

hf.close()
#hfB.close()
