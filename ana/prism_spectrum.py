#!/usr/bin/env python
"""
prism_spectrum.py : compare ggv-newton evts with PrismExpected
==================================================================

Scatter plot of wavelength against the appropriate prism coordinate
shows expected behaviour, except at the endpoints with outlier at high wavelength 
and a cliff drop off at low wavelenth. 
Perhaps these are due to restricted range of refractive index values or 
problem with spectral edges of the black body light source. 

TODO: check undispersed spot, extend refractive index values 

"""

import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt
from opticks.ana.boundary import Boundary   
from opticks.ana.tprism import Prism, PrismCheck, PrismExpected
from opticks.ana.cie  import cie_hist1d, cie_hist2d

deg = np.pi/180.


def spectrum_plot(w, d, db, ntile=50):
    
    hRGB_raw,hXYZ_raw, bx = cie_hist1d(w, d, db, colorspace="sRGB/D65", norm=2)

    hRGB_1 = np.clip(hRGB_raw, 0, 1)

    hRGB_2 = np.tile(hRGB_1, ntile ).reshape(-1,ntile,3)

    extent = [0,ntile,bx[-1],bx[0]] 

    ax.imshow(hRGB_2, origin="upper", extent=extent, alpha=1, vmin=0, vmax=1, aspect='auto')


def deviation_plot(w, d, db, ntile=50):

    h, hx = np.histogram(d, bins=db)   

    extent = [0,1,hx[-1],hx[0]] 

    ht = np.repeat(h,ntile).reshape(-1, ntile)

    im = ax.matshow(ht, origin="upper", extent=extent, alpha=1, aspect='auto')

    fig.colorbar(im)

    return ht


def uv_deviation_spike(d):
    """
    Deviation spike at 61 degrees
    coming from UV wavelengths all way from 60 to 340

    This is due to the refractive index high plateau 
    for GlassSchottF2 over that range...

    This invalidates the range, so cannot believe anything 
    below about 350nm

    :: 

        In [57]: n[d>60].min()
        Out[57]: 1.6774575260175819

        In [58]: n[d>60].max()
        Out[58]: 1.6846611499786377

    """
    plt.hist(d, bins=100)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = opticks_main(det="newton", tag="1", src="torch")

    plt.ion()

    try:
        sel = Evt(tag=args.tag, det=args.det, src=args.src, seqs=["TO BT BT SA"], args=args)  # newton, uses single incident angle
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    log.info("loaded %s " % repr(sel))

    boundary = Boundary("Vacuum///GlassSchottF2")

    prism = Prism("60.,300,300,0", boundary)

    n = boundary.imat.refractive_index(sel.wl)  

    xprism = PrismExpected(prism.a, n)

    pc = PrismCheck(prism, xprism, sel )

    w = pc.wx
    x = pc.p3[:,0]
    y = pc.p3[:,1]
    z = pc.p3[:,2]
    d = pc.dv/deg

    off = x != 700
    #assert np.all(x == 700.)

    fig = plt.figure()

    bands = [400,800]
    #bands = [w.min(),w.max()]

    nc = 2 
    nr = len(bands) - 1

    nb = 100 



    for ib in range(len(bands)-1):


        b = np.logical_and( w > bands[ib], w < bands[ib+1] )

        bins = np.linspace(d[b].min(), d[b].max(), nb )


        ax= fig.add_subplot(nr,nc,ib*2+1)

        spectrum_plot(w[b], d[b], bins)


        ax= fig.add_subplot(nr,nc,ib*2+2)

        deviation_plot(w[b], d[b], bins)


