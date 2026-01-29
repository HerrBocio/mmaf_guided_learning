epochs = [80,130,200]#,500]
#epochs = [700]
delta=.025
inp=3
h_t=[1]
archs=[[30,30,1],[100,100,1],[300,300,1]]#,[800,800,1]]
#archs = [[800,800,800,1]]  ##
if len(epochs) != len(archs):
    print("Epochs and archs have to have the same length.")
    epochs = epochs[:len(archs)]
datasetsM= ['Gaudiamonddata1A4mln','NIGdiamonddata1A4mln']#
datasetsname_short= ['Gau','NIG']#
m_batches=1000 
p=1
c=1
Ndraws=1000          

folder_day = "2201final_different_a_"
folder_day="1501final_"

pathT="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/tables/" #'results/s_data/tables/'
pathF="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/figures/" #'results/s_data/figures/'
#path='/afs/tu-chemnitz.de/project/calibration/paper_simulations/m_wise_epoch_wise/'

#path='/LOCAL/prol/s_data/'#nobias/'
path="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/"
day = "final_different_a"
day = "final_"