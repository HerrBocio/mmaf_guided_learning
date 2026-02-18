import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from matplotlib.ticker import LogLocator, LogFormatter, LogFormatterMathtext


# ----------------------------
# helper: spline in log-log
# ----------------------------
def loglog_spline(x, y, n=200, k=3):
    x_log = np.log10(x)
    y_log = np.log10(y)

    xnew_log = np.linspace(x_log.min(), x_log.max(), n)

    spl = make_interp_spline(x_log, y_log, k=k)
    ynew_log = spl(xnew_log)

    return 10**xnew_log, 10**ynew_log

def loglog_regression(x, y, n=200):
    x_log = np.log10(x)
    y_log = np.log10(y)

    # linear regression in log-log space
    coeff = np.polyfit(x_log, y_log, deg=3)
    poly = np.poly1d(coeff)

    xnew_log = np.linspace(x_log.min(), x_log.max(), n)
    ynew_log = poly(xnew_log)

    return 10**xnew_log, 10**ynew_log

#ORDER: GAU,NIG,OLR

datasets=['Gau','Nig','OLR']


# 	CRPS

MMAF=np.array([[],[]]) 
LSTM=np.array([[],[],[]])  
GRU=np.array([[[],[],[])   #  [0.0704,0.0563,0.0581,0.0562],[0.0407,0.0210,0.0267,0.0214]
DiffSTG=np.array([[],[],[]])

DiffSTG_emb = np.array([[],[],[]])





fig,axs = plt.subplots(2,3,figsize=(15, 8))


#bbox_to_anchor=(1.05, 1


axs[0,0].set_ylabel('CRPS')

axs[1,0].set_ylabel('RMSE',labelpad=32)





for i, idx in enumerate(datasets):
    
    # ----------------------------
    # DATA
    # ----------------------------
    x = np.array([10**2, 5*10**2, 10**3, 10**4, 10**5, 10**6])

    mmaf = MMAF[i, :]
    lstm = LSTM[i, :]
    gru  = GRU[i, :]
    diff = DiffSTG[i, :]
    diff_emb=DiffSTG_emb[i,:]

    # ----------------------------
    # CURVE SMOOTHING
    # ----------------------------
    x_mmaf_s, mmaf_s = loglog_spline(x, mmaf, n=300)

    mmaf_surrogate = np.array([0.3348,0.3336,0.3577])
    x_mmaf_surrogate,mmaf_surrogate = loglog_spline(x[:3], mmaf_surrogate, n=50,k=2)
    
    x_lstm_s, lstm_s = loglog_spline(x[:4], lstm, n=200)
    x_gru_s,  gru_s  = loglog_spline(x[:4], gru,  n=200)
    x_diff_s, diff_s = loglog_spline(x[2:], diff, n=200)
    x_diff_emb_s,diff_emb_s= loglog_spline(x[2:], diff_emb, n=200)
    
    # ----------------------------
    # PLOT LINES (smooth)
    # ----------------------------

    axs[0,i].plot(x[:4],lstm, color='orange', linestyle='--', linewidth=1.2, alpha=0.7)
    #plt.plot(x_lstm_s, lstm_s, color='orange', linestyle='--', linewidth=1.2, alpha=0.7)
    axs[0,i].plot(x_gru_s,  gru_s,  color='hotpink', linestyle='--', linewidth=1.2, alpha=0.7)
    axs[0,i].plot(x_diff_s, diff_s, color='forestgreen', linestyle='--', linewidth=1.2, alpha=0.7)
    #axs[0,i].plot(x_diff_emb_s, diff_emb_s, color='lightgreen', linestyle='--', linewidth=1.2, alpha=0.7)
    
    if idx=='OLR':
      axs[0,i].plot(x_mmaf_s, mmaf_s, color='royalblue', linewidth=1.6, alpha=0.85)
      plt.legend(
        [ 'ConvLSTM', 'ConvGRU', 'DiffSTG','MMAF'],
        loc='upper right',
        frameon=False
    )    
      #plt.plot(x_mmaf_surrogate[:36], mmaf_surrogate[:36], color='royalblue', linewidth=1.6, alpha=0.85)
      
      #plt.plot(x_mmaf_s[52:], mmaf_s[52:], color='royalblue', linewidth=1.6, alpha=0.85)
      #plt.plot(x_mmaf_s, mmaf_s, color='royalblue', linewidth=1.6, alpha=0.85)
    else:
      axs[0,i].plot(x_mmaf_s, mmaf_s, color='royalblue', linewidth=1.6, alpha=0.85)
      #plt.legend(
      #  [ 'ConvLSTM', 'ConvGRU', 'DiffSTG','Diff Embedded','MMAF'],
      #  loc='upper right',
      #  frameon=False
    #)
    # ----------------------------
    # PLOT POINTS
    # ----------------------------
    axs[0,i].scatter(
        x, mmaf,
        marker='>',
        s=40,
        color='royalblue',
        zorder=5
    )
    
    axs[0,i].scatter(
        x[:4], lstm,
        marker='D',
        s=25,
        color='orange',
        alpha=0.9,
        zorder=4
    )
    
    axs[0,i].scatter(
        x[:4], gru,
        marker='D',
        s=25,
        color='hotpink',
        alpha=0.9,
        zorder=4
    )
    
    axs[0,i].scatter(
        x[2:], diff,
        marker='D',
        s=25,
        color='forestgreen',
        alpha=0.9,
        zorder=4
    )
    '''
    axs[0,i].scatter(
        x[2:], diff_e,,
        marker='D',
        s=25,
        color='lightgreen',
        alpha=0.9,
        zorder=4
    )
    '''
    # ----------------------------
    # AXES (log + populated Y)
    # ----------------------------
    axs[0,i].set_xscale('log')
    axs[0,i].set_yscale('log')
    axs[0,i].set_title(idx,fontsize=12)
    #ax = axs[0,i].gca()
    
    # Limiti asse Y (fino a 2×10^-1)
    axs[0,i].set_ylim(bottom=None, top=np.max([np.max(mmaf),np.max(lstm),np.max(gru),np.min(diff)])+.05)
    
    # Major ticks (10^n) → mostrati come 10^{-1}, 10^{-2}, ...
    #ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    #ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    
    # Minor ticks (2–9 × 10^n)
    #ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.linspace(2.0, 10, 10, endpoint=False)))
    #ax.yaxis.set_minor_locator( LogLocator(base=10.0, subs=np.arange(10, 60, 2) * 0.005) )
    
    axs[0,i].tick_params(axis='y', which='minor', length=3)
    

##################################################################
	
#	RMSE



MMAF=np.array([[],[],[]])
LSTM=np.array([[],[],[]]) 
GRU=np.array([[],[],[]])   
DiffSTG=np.array([[],[],[]])

DiffSTG_emb= np.array([[],[],[]])

for i, idx in enumerate(datasets):
    
    # ----------------------------
    # DATA
    # ----------------------------
    x = np.array([10**2, 5*10**2, 10**3, 10**4, 10**5, 10**6])

    mmaf = MMAF[i, :]
    lstm = LSTM[i, :]
    gru  = GRU[i, :]
    diff = DiffSTG[i, :]
    diff_emb=DiffSTG_emb[i,:]


    # ----------------------------
    # CURVE SMOOTHING
    # ----------------------------
    x_mmaf_s, mmaf_s = loglog_spline(x, mmaf, n=300)

    mmaf_surrogate = np.array([0.3348,0.3336,0.3577])
    x_mmaf_surrogate,mmaf_surrogate = loglog_spline(x[:3], mmaf_surrogate, n=50,k=2)
    
    x_lstm_s, lstm_s = loglog_spline(x[:4], lstm, n=200)
    x_gru_s,  gru_s  = loglog_spline(x[:4], gru,  n=200)
    x_diff_s, diff_s = loglog_spline(x[2:], diff, n=200)
    x_diff_emb_s,diff_emb_s= loglog_spline(x[2:], diff_emb, n=200)
    
    # ----------------------------
    # PLOT LINES (smooth)
    # ----------------------------

    
    axs[1,i].plot(x[:4], lstm, color='orange', linestyle='--', linewidth=1.2, alpha=0.7)
    axs[1,i].plot(x_gru_s,  gru_s,  color='hotpink', linestyle='--', linewidth=1.2, alpha=0.7)
    axs[1,i].plot(x_diff_s, diff_s, color='forestgreen', linestyle='--', linewidth=1.2, alpha=0.7)
    #axs[1,i].plot(x_diff_emb_s, diff_emb_s, color='lightgreen', linestyle='--', linewidth=1.2, alpha=0.7)
    #if idx=='OLR':
      #plt.plot(x_mmaf_surrogate[:36], mmaf_surrogate[:36], color='royalblue', linewidth=1.6, alpha=0.85)

      #plt.plot(x_mmaf_s[52:], mmaf_s[52:], color='royalblue', linewidth=1.6, alpha=0.85)
    
      #plt.plot(x_mmaf_s, mmaf_s, color='royalblue', linewidth=1.6, alpha=0.85)
    #else:
    #  axs[1,i].plot(x_mmaf_s, mmaf_s, color='royalblue', linewidth=1.6, alpha=0.85)

    axs[1,i].plot(x_mmaf_s, mmaf_s, color='royalblue', linewidth=1.6, alpha=0.85)

    # ----------------------------
    # PLOT POINTS
    # ----------------------------
    axs[1,i].scatter(
        x, mmaf,
        marker='>',
        s=40,
        color='royalblue',
        zorder=5
    )
    
    axs[1,i].scatter(
        x[:4], lstm,
        marker='D',
        s=25,
        color='orange',
        alpha=0.9,
        zorder=4
    )
    
    axs[1,i].scatter(
        x[:4], gru,
        marker='D',
        s=25,
        color='hotpink',
        alpha=0.9,
        zorder=4
    )
    
    axs[1,i].scatter(
        x[2:], diff,
        marker='D',
        s=25,
        color='forestgreen',
        alpha=0.9,
        zorder=4
    )
  
    '''
    axs[1,i].scatter(
        x[2:], diff_emb,
        marker='D',
        s=25,
        color='lightgreen',
        alpha=0.9,
        zorder=4
    )
    '''
#
    
    # ----------------------------
    # AXES (log + populated Y)
    # ----------------------------
    axs[1,i].set_xscale('log')
    #axs[1,i].set_yscale('log')
    #axs[1,i].set_title(idx,fontsize=12)
    #ax = axs[1,i].gca()
    
    # Limiti asse Y (fino a 2×10^-1)
    axs[1,i].set_ylim(bottom=None, top=np.max([np.max(mmaf),np.max(lstm),np.max(gru),np.min(diff)])+.05)
    
    # Major ticks (10^n) → mostrati come 10^{-1}, 10^{-2}, ...
    #ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    #ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    
    # Minor ticks (2–9 × 10^n)
    #ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.linspace(2.0, 10, 10, endpoint=False)))
    #ax.yaxis.set_minor_locator( LogLocator(base=10.0, subs=np.arange(10, 60, 2) * 0.005) )
    
    axs[1,i].tick_params(axis='y', which='minor', length=3)
# ----------------------------
# LABELS & TITLE
# ----------------------------
axs[1,1].set_xlabel("Number of Parameters", fontsize=11)

# ----------------------------
# LEGEND & SAVE
# ----------------------------
fig.legend(
        ['ConvLSTM', 'ConvGRU', 'DiffSTG','MMAF'],
        loc='upper center',
        #loc= (0.5,0.6),
        ncols = 4,
        fontsize=14,
        #bbox_to_anchor=(1.05, 1),
        frameon=True
    )

fig.subplots_adjust(left= 0.08,right=.97)
#plt.tight_layout()
plt.savefig("spline_siam.jpg")
plt.savefig("spline_siam.eps",format='eps')
plt.close()




######################################## FLOPs

#	SPECIFICALLY DESIGNED FOR OLR


MMAF=np.array([])/36
LSTM=np.array([])*71
GRU=np.array([])*71
DiffSTG=np.array([])*2340

#([859880000,12296960000,90685480000,90685480000])

fig,axs = plt.subplots(1,1,figsize=(6, 6))

# ----------------------------
# DATA
# ----------------------------
x = np.array([10**2, 5*10**2, 10**3, 10**4, 10**5, 10**6])

mmaf = MMAF
lstm = LSTM
gru  = GRU
diff = DiffSTG

# ----------------------------
# CURVE SMOOTHING
# ----------------------------
x_mmaf_s, mmaf_s = loglog_spline(x, mmaf, n=300)

x_lstm_s, lstm_s = loglog_spline(x[:4], lstm, n=200)
x_gru_s,  gru_s  = loglog_spline(x[:4], gru,  n=200)
x_diff_s, diff_s = loglog_spline(x[2:], diff, n=200)

# ----------------------------
# PLOT LINES (smooth)
# ----------------------------

axs.plot(x[:4],lstm, color='orange', linestyle='--', linewidth=1.2, alpha=0.7)
axs.plot(x_gru_s,  gru_s,  color='hotpink', linestyle='--', linewidth=1.2, alpha=0.7)
axs.plot(x_diff_s, diff_s, color='forestgreen', linestyle='--', linewidth=1.2, alpha=0.7)
axs.plot(x_mmaf_s, mmaf_s, color='royalblue', linewidth=1.6, alpha=0.85)

# ----------------------------
# PLOT POINTS
# ----------------------------
axs.scatter(
    x, mmaf,
    marker='>',
    s=40,
    color='royalblue',
    zorder=5
)

axs.scatter(
    x[:4], lstm,
    marker='D',
    s=25,
    color='orange',
    alpha=0.9,
    zorder=4
)

axs.scatter(
    x[:4], gru,
    marker='D',
    s=25,
    color='hotpink',
    alpha=0.9,
    zorder=4
)

axs.scatter(
    x[2:], diff,
    marker='D',
    s=25,
    color='forestgreen',
    alpha=0.9,
    zorder=4
)

# ----------------------------
# AXES (log + populated Y)
# ----------------------------
axs.set_xscale('log')
axs.set_yscale('log')

# Limiti asse Y (fino a 2×10^-1)
axs.set_ylim(bottom=None, top=np.max([np.max(mmaf),np.max(lstm),np.max(gru),np.max(diff)])*10)

axs.tick_params(axis='y', which='minor', length=3)

axs.set_xlabel('Parameters')

axs.set_ylabel('FLOPs')

fig.legend(
        ['ConvLSTM', 'ConvGRU', 'DiffSTG','MMAF'],
        loc='upper center',
        #loc= (0.5,0.6),
        ncols = 4,
        fontsize=11,
        #bbox_to_anchor=(1.05, 1),
        frameon=True
        )

fig.subplots_adjust(left=0.11,right=.97,bottom=0.1)
#plt.tight_layout()
plt.savefig("spline_siam_flops.png")
plt.savefig("spline_siam_flops.eps",format='eps')
plt.close()
