import netCDF4 as nc


delta=.025
inp=3
h_t=[1]
archs=[[30,30,1],[100,100,1],[300,300,1]]
epochs = [30000,45000,90000]


file_path = '/afs/tu-chemnitz.de/project/calibration/OLR_full.nc'#Almut_plusFuture.nc'
#olr = nc.Dataset(file_path, mode="r").variables
#print(olr)
#data=olr['olra'][:,:]#,:]
datasetsname_short= ['OLR']#
p=1
c=1
Ndraws=1000          

folder_day = "0202_"

pathT="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/tables/" #'results/s_data/tables/'
pathF="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/figures/" #'results/s_data/figures/'
#path='/afs/tu-chemnitz.de/project/calibration/paper_simulations/m_wise_epoch_wise/'

#path='/LOCAL/prol/s_data/'#nobias/'
path="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/"
pathsave="/afs/tu-chemnitz.de/project/mmaf_unbounded/"+folder_day+"results/"
day = "0202"
piScalingLabel=list(range(10,230,20))
