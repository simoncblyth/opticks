#!/usr/bin/env python
"""

Kurt Nassau, The Physics and Chemistry of Color

p219, Dispersion with single reflection (in spherical raindrop)
      (shadow of head at center of bow)

first bow
      V:41 R:43 degree cone around the eye-antisolar point axis

2nd bow 
      R:50, V:54 

Alexanders dark band, between the bows (due to no rays below min deviation)

First 15 bows... 


* http://www.patarnott.com/atms749/pdf/MultipleRainbowsSingleDrops.pdf
* 


"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from env.numerics.npy.ana import Evt, Selection, costheta_
from env.numerics.npy.geometry import Boundary   

from env.numerics.npy.cie  import cie_hist1d, cie_hist2d


def rainbow(ax, pos, w, nb=100, rmin=-1, mode="cie", style="1d"):

    assert style in ["cie", "hist", "test"]
    assert mode in ["1d","2d"]

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    r = np.sqrt(y*y + z*z)
    #assert np.all(x == -700. )
    # huh 


    yb = np.linspace(y.min(), y.max(), nb)
    zb = np.linspace(z.min(), z.max(), nb)
    rb = np.linspace(r.min(), r.max(), nb)
    ntile = nb
    tb = np.linspace(0, 1, ntile)

    b = r > rmin

    if style == "cie":

        if mode == "1d":
            hRGB_raw, hXYZ_raw, bx= cie_hist1d(w[b],r[b], rb, colorspace="sRGB/D65")
            hRGB_1d = np.clip(hRGB_raw, 0, 1)
            hRGB = np.tile(hRGB_1d, ntile ).reshape(-1,ntile,3)
            extent = [0,ntile,bx[0],bx[-1]] 
        elif mode == "2d":
            hRGB_raw, hXYZ_raw, extent = cie_hist2d(w[b],y[b],z[b], yb, zb, colorspace="sRGB/D65")
            hRGB = np.clip(hRGB_raw, 0, 1)
        pass
        ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1)

    elif style == "hist":

        if mode == "2d":

            _,_,_,im = ax.hist2d(y[b],z[b], bins=[yb,zb]) 
            fig.colorbar(im)

        elif mode == "1d":

            h, hx = np.histogram(r[b],bins=rb)   
            extent = [0,ntile,hx[0],hx[-1]] 
            ht = np.repeat(h,ntile).reshape(-1, ntile)
            im = ax.matshow(ht, origin="lower", extent=extent, alpha=1)
            fig.colorbar(im)
   
    else:
        pass
        

    return r



def rainbow_incident_angle(n, k=1):
    """
    Incident angle of minimum deviation for k reflection rainbow
    """
    return np.arccos( np.sqrt((n*n - 1.)/(k*(k+2.)) ))

def rainbow_deviation_angle(n, k=1):
    """
    Total deviation angle at minimum deviation
    """
    i = rainbow_incident_angle(n, k)
    r = np.arcsin(np.sin(i)/n)
    return k*np.pi + 2*i - 2*r*(k+1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    plt.ion()

    evt = Evt(tag="1", det="rainbow")

    sel = Selection(evt,"BT BR BT SA")  # single reflection corresponds to 1st rainbow

    boundary = Boundary("Vacuum///MainH2OHale")
    #boundary = Boundary("Vacuum///GlassSchottF2")

    n = boundary.imat.refractive_index(sel.wl)  
    wd = np.arange(80,820,10)
    nd = boundary.imat.refractive_index(wd)  

    w = sel.recwavelength(0)  # should be same for all records 

    p0 = sel.recpost(0)
    p1 = sel.recpost(1)
    p2 = sel.recpost(2)
    p3 = sel.recpost(3)
    p4 = sel.recpost(4)

    p01 = p1 - p0   # incident ray 
    p34 = p4 - p3   # exitant ray 
    cdv = costheta_(p01, p34)

    dv = np.arccos(cdv)                 # simulated deviation angle
    xv = rainbow_deviation_angle(n, 1)  # expected deviation angle

    #plt.hist((xv - dv)*180./np.pi, bins=100)  
    #spikes at zero, but asymmetric tail off to -13 degrees
    #plt.hist(dv*180./np.pi, bins=100)
    #
    #plt.hist2d(dv*180./np.pi, w, bins=100)
    #
    # kink in deviation against wavelength at 330 nm for both simulated and expected
    # suggests its an issue with the refractive index  
    # YEP: see same shape in refractive index alone, 
    #      plt.close(); plt.hist2d( n, w, bins=100)
    #
    # Plateau in refractive index below 330nm, probably edge of data artifact ? 
    #   plt.close();plt.plot(wd, nd)
    #

    x = p4[:,0]
    y = p4[:,1]
    z = p4[:,2]
    r = np.sqrt(y*y + z*z)


    fig = plt.figure()


    nb = 100
    rmin = 40
    mode = "2d"

    ax= fig.add_subplot(1,2,1)
    rainbow(ax, p4, w, nb=nb, rmin=rmin, style="cie", mode=mode )

    ax= fig.add_subplot(1,2,2)
    rainbow(ax, p4, w, nb=nb, rmin=rmin, style="hist", mode=mode )



