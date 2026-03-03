import numpy as np
from scipy.io import loadmat
import os
#from tqdm import trange
def get_simulated_data(filename):
    data=loadmat(filename+'.mat')
    data=data["data"]
    return data

def create_folder(new_path):
  '''
  Creates directory 'new_path'
  '''
  if not os.path.exists(new_path):
    os.makedirs(new_path)
    print("folder created")



datasetsM= ['Gaudiamonddata1A4mln','NIGdiamonddata1A4mln']#

data_path='../Desktop/ffnn/datasets/'

path='dataset/'


for current_id in range(len(datasetsM)):
	
	data=get_simulated_data(data_path+datasetsM[current_id] ) #might be converted into JNP
	create_folder(path+datasetsM[current_id])
	data=data[:10,:]
	np.save(path+datasetsM[current_id]+''+'/'+'data.npy',data)
	