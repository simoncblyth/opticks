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

"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from env.numerics.npy.ana import Evt, Selection
from env.numerics.npy.geometry import Boundary   

from env.numerics.npy.cie  import cie_hist1d, cie_hist2d


def rainbow(pos,w,onedim=True):
    x = pos[:,0]
    assert np.all(x == -700. )
    y = pos[:,1]
    z = pos[:,2]
    r = np.sqrt(y*y + z*z)
    b = r > 10

    if onedim:
        ntile = 500 
        hRGB_raw, hXYZ_raw, bx= cie_hist1d(w[b],r[b], nx=100, colorspace="sRGB/D65")
        hRGB_1d = np.clip(hRGB_raw, 0, 1)
        hRGB = np.tile(hRGB_1d, ntile ).reshape(-1,ntile,3)
        extent = [0,ntile,bx[0],bx[-1]] 
    else:
        hRGB_raw, hXYZ_raw, extent = cie_hist2d(w[b],y[b],z[b], nx=100, ny=100, colorspace="sRGB/D65")
        hRGB = np.clip(hRGB_raw, 0, 1)
    pass
    ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    plt.ion()

    evt = Evt(tag="1", det="rainbow")

    sel = Selection(evt,"BT BR BT SA")  

    boundary = Boundary("Vacuum///GlassSchottF2")

    n = boundary.imat.refractive_index(sel.wl)  

    wx = sel.recwavelength(0)  # should be same for all records 
    p4 = sel.recpost(4)

    fig = plt.figure()

    ax= fig.add_subplot(1,1,1)

    rainbow( p4, wx, onedim=True )



