
epochs = [130]

delta=.025
inp=3
h_t=[1]
archs=[[100,100,1]]
if len(epochs) != len(archs):
    print("Epochs and archs have to have the same length.")
    epochs = epochs[:len(archs)]
datasetsM= ['Gaudiamonddata1A4mln']#
datasetsname_short= ['Gau','NIG']#
m_batches=1000 
p=1
c=1
Ndraws=1000          

#folder_day = "2201final_different_a_"
#folder_day="1501final_"
folder_day ="2901__"

pathT="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/tables/" #'results/s_data/tables/'
pathF="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/figures/" #'results/s_data/figures/'
#path='/afs/tu-chemnitz.de/project/calibration/paper_simulations/m_wise_epoch_wise/'

#path='/LOCAL/prol/s_data/'#nobias/'
path="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/"
#day = "final_different_a"
#day = "final_"
day = "2901"

do_f_epochs = False
save_eps = False
