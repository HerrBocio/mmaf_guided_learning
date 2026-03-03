


def pit_value(ef,b):
  x_vals = ef.transpose()     
  b_expanded = b_vals[:, np.newaxis]  # shape: (N, 1)
  pitMap = (b_expanded > x_vals).astype(np.int16)
  pitMap = np.sum(pitMap, axis=1)
  print(pitMap)


    
def plot(metrics)  
    pathF='metrics/figures/'
    list_coords = metrics.data_val.shape[0]
    pathPrior=pathF+metrics.filename+'/EF/'+'prior'+str(int(1./pi))+'var/'+day+'/'+str(metrics.model.dim)+'/'
    print('mm',m_e_f.shape,b_test.shape)
    pit_single=lambda x,b: (b<=x)*0 + (b>x)*1
    
    qs_label=['10%','20%','30%','40%','50%','60%','70%','80%','90%','95%']#print(crps)
    for Coord in list_coords:

      
      fig,ax=plt.subplots(figsize=[12,5])
      #for cone_idx, his in enumerate(fig.axes):
      ax.hist(pitMap, histtype="bar",cumulative=False,density=True,align='mid')
      #ax.set_axis_off()
      plt.tight_layout()
      plt.savefig(pathPrior+'mmaf_hist'+str(Coord)+'_'+datasetsM[current_id]+'.eps',format='eps')
     
      plt.savefig(pathPrior+'mmaf_hist'+str(Coord)+'_'+datasetsM[current_id]+'.png')
      plt.close()
    
      
      
      fig, ax = plt.subplots(figsize=(12,5))
    
      cmap = plt.cm.PuBu  # define the colormap
      # extract all colors from the .jet map
      cmaplist = [cmap(n3) for n3 in range(cmap.N)]
      #print(len(cmaplist))#
       # force the first color entry to be grey
      #cmaplist[0] = (.5, .5, .5, 1.0)
      
      # create the new map
      #cmap = mpl.colors.LinearSegmentedColormap.from_list(
      #    'Custom cmap', cmaplist, cmap.N)
      #plt.figure(figsize=(16,20))                  
      
        
      #fig, axs = plt.subplots(4, 2,figsize=(8,6))
      #plt.subplots_adjust(hspace=0)
      fig.tight_layout()
    
      #for n4, ax in enumerate(fig.axes):
        #ax.set_ylabel(str(i))
      for n5,el in enumerate(qs):
      
          #start_date = datetime.date(2023, 1, 1)
          #step_days = 10  # ogni step è di 10 giorni
          
          # Genera asse x in formato datetime
          n_steps_model = m_e_f[:, :, :Ncrps].shape[-1]
          n_steps_test = b_test[:, :Ncrps].shape[-1]
          
          ax.grid(True, axis='x', which='major', linestyle='-', linewidth=0.6)
          ax.grid(True, axis='y', linestyle='-', linewidth=0.6)
          # Rotate x-tick labels if crowded
          # Fill between quantiles
          ax.fill_between(
              range(1,Ncrps+1),
              np.quantile(m_e_f[Coord, :, :Ncrps], q=el, axis=0),
              np.quantile(m_e_f[Coord, :, :Ncrps], q=1-el, axis=0),
              color=cmaplist[60 + 15 * (n5 + 1)]
          )
          
          ax.plot(range(1,Ncrps+1), b_test[Coord, :Ncrps], color='gold', linewidth=0.5)
          
          # Formatting
          ax.set_ylim(-.7, .7)
          ax.set_xlabel('Forecasting Horizons',fontsize=21)
        
          # Limiti dell'asse x: inizia esattamente da gennaio
          #ax.set_xlim([start_date, x_values_model[-1]])
          
          # Griglia
          
        #ax.annotate('average crps: '+ str(np.round(np.mean(crps[2*i],axis=0),decimals=4)) ,xy=(30,np.amax(np.quantile(m_e_f[2*i,:,:],q=.975,axis=0))-.01),fontsize='x-small')
      
      
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
      plt.legend(custom_lines,reversed([el for el in qs_label]),labelspacing=1.5, ncol=1,bbox_to_anchor=(1, .99))#, loc="upper left")
    #plt.colorbar(cmaplist[-1:-20*(j+1):-20])
      plt.savefig(pathPrior+str(Coord)+'_time_EF?'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.jpg',bbox_inches='tight')
      plt.savefig(pathPrior+str(Coord)+'_time_EF?'+figure_name+str(dim([inp,*arch])) +'_' +str(datasetsM[current_id])+'pir'+str(pir)+'.eps',bbox_inches='tight',format='eps')
    
      plt.close()
    
                            