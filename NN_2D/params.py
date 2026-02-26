

delta=.025
inp=3
h_t=[1]
#epochs = [12,25,50,75,80]
epochs = [75]
#archs=[[10,10,1],[30,30,1],[100,100,1],[300,300,1],[10,10,10,10,10,1]]
archs=[[300,300,1]]
if len(epochs) != len(archs):
    print("Epochs and archs have to have the same length.")
    epochs = epochs[:len(archs)]
datasetsM= ['Gaudiamonddata1A4mln','NIGdiamonddata1A4mln']#
datasetsname_short= ['Gau','NIG']#
m_batches=1000 
p=1
c=1
Ndraws=1000          

#folder_day = "2201final_different_a_"
#folder_day="1501final_"
folder_day ="2002_"

pathT="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/tables/" #'results/s_data/tables/'
pathF="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/figures/" #'results/s_data/figures/'
#path='/afs/tu-chemnitz.de/project/calibration/paper_simulations/m_wise_epoch_wise/'

#path='/LOCAL/prol/s_data/'#nobias/'
path="/afs/tu-chemnitz.de/home/urz/j/jasst/"+folder_day+"results/"
#day = "final_different_a"
#day = "final_"
day = "2002"
pathsave='/LOCAL/jasst/2002_results/'

do_f_epochs =True
save_eps = False
piRescaling=list(range(10,230,20))


bounded_Gau = [" & N(0,1/30) & 0.0598 & 0.017 \\",
 "& N(0,1/30) & 0.0633 & 0.0315 \\",
 "& N(0,1/90) & 0.058 & 0.0181 \\",
 "& N(0,1/110) & 0.0711 & 0.0418 \\",
 "& N(0,1/30) & 0.0609 & 0.0162\\"]

bounded_Nig = [r"& N(0,1/210) & 0.0219 & 0.0019\\",
 r"& N(0,1/210) & 0.0211 & 0.002 \\",
 r"& N(0,1/210) & 0.0219 & 0.0029 \\",
 r"& N(0,1/210) & 0.0297 & 0.0079 \\",
 r"& N(0,1/210) & 0.0217 & 0.0019\\"]