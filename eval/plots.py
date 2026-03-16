import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import datetime
import os
from numpy import quantile
from jax.numpy import sum
from jax import vmap
from src.utils import create_folder
from scipy.stats import  chisquare,kstest
from scipy.stats import randint
import numpy as np

def pit_value(ef,b):
  q=lambda x: (b<=x)*0 + (b>x)*1
  s=vmap(q)(ef)
  s=sum(s)
  return s

    
def mmaf_plot(metrics,pi,width,depth,min_error,epochs): 
    pathF='metrics/figures/'
    list_coords = range(metrics.data_val.shape[0])
    pathPrior=pathF+metrics.filename+'/ef/'+'[' + str(width) +'^'+str(depth)+ ']'+'/'
    create_folder(pathPrior)

    qs=[.025,.05,.1,.15,.2,.25,.3,.35,.4,.45]
    qs_label=['10%','20%','30%','40%','50%','60%','70%','80%','90%','95%']
    for Coord in list_coords:
      
      pitMap=vmap(pit_value,in_axes=(1,0))(metrics.ef_test[Coord,:,:],metrics.data_test[Coord,-1,:])
      #print(pitMap)
      Xtesting= chisquare(f_obs=pitMap)#.pvalue
      kstesting = kstest(pitMap,randint.rvs(low=0,high=1000,size=100)).pvalue 
      #print('\n\n\t p-val \t chisq \t ks\n\t\t',Xtesting,'\t',kstesting)
            
      fig,ax=plt.subplots(figsize=[12,5])
      ax.hist(pitMap, histtype="bar",cumulative=False,density=True,align='mid')
      plt.tight_layout()
      plt.savefig(pathPrior+'mmaf_hist_'+ metrics.filename + '_[' + str(width) +'^'+str(depth)+ ']_'+ str(pi)+'_'+str(8)+'_'+str(Coord)+'.eps',format='eps')
      plt.savefig(pathPrior+'mmaf_hist_'+ metrics.filename + '_[' + str(width) +'^'+str(depth)+ ']_'+ str(pi)+'_'+str(8)+'_'+str(Coord)+'.png')
      plt.close()
      
      fig, ax = plt.subplots(figsize=(12,5))
      cmap = plt.cm.PuBu 
      cmaplist = [cmap(el) for el in range(cmap.N)]
      fig.tight_layout()
      for el_idx,el in enumerate(qs):
          n_steps_model = metrics.ef_test.shape[-1]
          n_steps_test = metrics.data_test.shape[-1]
          
          ax.grid(True, axis='x', which='major', linestyle='-', linewidth=0.6)
          ax.grid(True, axis='y', linestyle='-', linewidth=0.6)
          ax.fill_between(
              range(1,metrics.ef_test.shape[-1]+1),
              quantile(metrics.ef_test[Coord,:,:], q=el, axis=0),
              quantile(metrics.ef_test[Coord,:,:], q=1-el, axis=0),
              color=cmaplist[60 + 15 * (el_idx + 1)]
          )
          
          ax.plot(range(1,metrics.data_test.shape[-1]+1), metrics.data_test[Coord,-1,:], color='gold', linewidth=0.5)
          #ax.set_ylim(-.7, .7)
          ax.set_xlabel('Forecasting Horizons',fontsize=21)
        
      custom_lines = [Line2D([0], [0],marker='s', color=cmaplist[60+15*(0+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(1+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(2+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(3+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(4+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(5+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(6+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(7+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(8+1)], lw=8),
                      Line2D([0], [0],marker='s', color=cmaplist[60+15*(9+1)], lw=8),
                     ]
      plt.legend(custom_lines,reversed([el for el in qs_label]),labelspacing=1.5, ncol=1,bbox_to_anchor=(1, .99))
      plt.savefig(pathPrior + 'mmaf_ef_' + metrics.filename + '_[' + str(width) +'^'+str(depth)+ ']_'+  str(pi)+'_'+str(8)+'_'+str(Coord)+'.jpg',bbox_inches='tight')
      plt.savefig(pathPrior + 'mmaf_ef_' + metrics.filename + '_[' + str(width) +'^'+str(depth)+ ']_'+  str(pi)+'_'+str(8)+'_'+str(Coord)+'.eps',bbox_inches='tight',format='eps')
      plt.close()

    
    plt.figure(figsize=(6, 3))                  
    #plt.subplot(311)
    plt.yscale("log")
    
    plt.plot(range(1,epochs+1),min_error[0][:,:,0].mean(axis=1)+min_error[1][:,:,0].mean(axis=1),linewidth=1,color='red')

    plt.xlabel("epochs")
    #plt.xticks(range(Nbatches*10,len(bound_train)*Nbatches+1,Nbatches*10),list(range(10, len(bound_train)+1,10)))

    plt.legend(['Target function','MC Target function'])#,'vacuity threshold'])
    plt.tight_layout()
    plt.savefig(pathPrior + 'mmaf_ef_overlap' + metrics.filename + '_[' + str(width) +'^'+str(depth)+ ']_'+  str(pi)+'_'+str(8)+'_'+'.png')

    plt.savefig(pathPrior + 'mmaf_ef_overlap' + metrics.filename + '_[' + str(width) +'^'+str(depth)+ ']_'+  str(pi)+'_'+str(8)+'_'+'.eps',format='eps')
    
    plt.close()
    
                            
