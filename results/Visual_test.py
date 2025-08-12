import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('Prova_Pratica2025-08-08 12:42:27.726833.h5', 'r') as f:
    gruppo = f['LeoLori']
    minimo = gruppo['min'][()]
    massimo = gruppo['max'][()]
    test = gruppo['test'][()]
    q25 = gruppo['q25'][()]
    q75 = gruppo['q75'][()]
    c = gruppo['c'][()]
    p = gruppo['p'][()]


    valmin = np.min([np.nanmin(q25[0,:,:]),np.nanmin(test),np.nanmin(q75[0,:,:])])
    valmax = np.max([np.nanmax(q25[0,:,:]),np.nanmax(test),np.nanmax(q75[0,:,:])])

    
    # q25
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(q25[0,:,:], cmap='inferno', vmin=valmin , vmax=valmax) 
    ax1.set_title('25 quantile')
    fig1.colorbar(im1, ax=ax1)
    fig1.savefig("OLR25quantile.png")
    plt.close(fig1)

    # test
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(test, cmap='inferno', vmin=valmin , vmax=valmax)
    ax2.set_title('test set')
    fig2.colorbar(im2, ax=ax2)
    fig2.savefig("OLRtest.png")
    plt.close(fig2)

    # q75
    fig3, ax3 = plt.subplots()
    im3 = ax3.imshow(q75[0,:,:], cmap='inferno', vmin=valmin , vmax=valmax)
    ax3.set_title('75 quantile')
    fig3.colorbar(im3, ax=ax3)
    fig3.savefig("OLR75quantile.png")
    plt.close(fig3)


    pixels=np.array([])
    out_q25=np.array([])
    out_q75=np.array([])
    ground=np.array([])
    count=0

    for i in range(len(q25[0,0,:])):
        for j in range(len(q25[0,:,0])):
            if (i < c*p or j < c*p) or (i > len(q25[0,0,:])-1-c*p or j > len(q25[0,:,0])-1-c*p):
                continue
            else:
                out_q25=np.append(out_q25, q25[0,j,i])
                out_q75=np.append(out_q75, q75[0,j,i])
                ground=np.append(ground, test[j,i])
                pixels=np.append(pixels, [count])
                count = count+1
            

    fig4, ax4 = plt.subplots()
    ax4.plot(pixels, out_q25, 'r') 
    ax4.plot(pixels, ground, 'g') 
    ax4.plot(pixels, out_q75, 'r')
    ax4.set_title('lexicographic order')
    fig4.savefig("OLRlexicographic.png")
    plt.close(fig4)


        

    