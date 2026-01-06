epochs_nopreT = 60
delta=.025
inp=3
h_t=[1]
archs=[[30,30,1]]  ##
datasetsM= ['Gaudiamonddata1A4mln','NIGdiamonddata1A4mln']#
datasetsname_short= ['Gau','NIG']#
m_batches=1000 
p=1
c=1
Ndraws=1000          

folder_day = "0601"

pathT="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/tables/" #'results/s_data/tables/'
pathF="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/figures/" #'results/s_data/figures/'
#path='/afs/tu-chemnitz.de/project/calibration/paper_simulations/m_wise_epoch_wise/'

#path='/LOCAL/prol/s_data/'#nobias/'
path="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/"
day = folder_day
day='0601'
