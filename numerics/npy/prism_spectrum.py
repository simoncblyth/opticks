#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from env.numerics.npy.ana import Evt, Selection
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.prism import Prism, PrismCheck, PrismExpected
from env.numerics.npy.cie  import cie_hist1d, cie_hist2d


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    plt.ion()

    evt = Evt(tag="1", det="prism")

    sel = Selection(evt,"BT BT SA")  

    boundary = Boundary("Vacuum///GlassSchottF2")

    prism = Prism("60.,300,300,0", boundary)

    n = boundary.imat.refractive_index(sel.wl)  

    xprism = PrismExpected(prism.a, n)

    pc = PrismCheck(prism, xprism, sel )

    w = pc.wx
    x = pc.p3[:,0]
    y = pc.p3[:,1]
    z = pc.p3[:,2]
    assert np.all(x == 700.)

    fig = plt.figure()

    #bands = [300,800]
    bands = [w.min(),w.max()]

    nc = len(bands) - 1
    nr = 1 
    ntile = 50


    for ib in range(len(bands)-1):

        ax= fig.add_subplot(nr,nc,ib+1)

        b = np.logical_and( w > bands[ib], w < bands[ib+1] )

        nx = 100 

        hRGB_raw,hXYZ_raw, bx = cie_hist1d(w[b],y[b], nx=nx, colorspace="sRGB/D65")

        # uniform scaling by maximal single colors, prior to clipping 
        # gets rid of the unphysical color repetition
        # but theres kinda a gap between the green and the blue 

        #hRGB_raw /= hRGB_raw[0,:,0].max()  # scaling by maximal red, results in muted spectrum
        #hRGB_raw /= hRGB_raw[0,:,1].max()  # scaling by maximal green,  OK  
        #hRGB_raw /= hRGB_raw[0,:,2].max()  # scaling by maximal blue, similar to green by pumps the blues and nice yellow  
    

        hRGB_1 = np.clip(hRGB_raw, 0, 1)

        hRGB_2 = np.tile(hRGB_1, ntile ).reshape(-1,ntile,3)

        extent = [0,ntile,bx[0],bx[-1]] 

        ax.imshow(hRGB_2, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1)

        #ax.scatter(y[b][::10],w[b][::10],c=hRGB_1[0][::10])




