epochs_nopreT = 50
delta=.025
inp=3
h_t=[1]
archs=[[10,10,1],[30,30,1],[100,100,1],[300,300,1]]  ##,,[300,300,1],[100,100,1],
datasetsM= ['Gaudiamonddata1A4mln','NIGdiamonddata1A4mln']#
m_batches=1000 
p=1
c=1
Ndraws=1000          


pathT="/afs/tu-chemnitz.de/home/urz/j/jasst/n1712results/tables/" #'results/s_data/tables/'
pathF="/afs/tu-chemnitz.de/home/urz/j/jasst/n1712results/figures/" #'results/s_data/figures/'
#path='/afs/tu-chemnitz.de/project/calibration/paper_simulations/m_wise_epoch_wise/'

#path='/LOCAL/prol/s_data/'#nobias/'
path="/afs/tu-chemnitz.de/home/urz/j/jasst/n1712results/nopreT/"
day = '151225151225'#day='1512'
