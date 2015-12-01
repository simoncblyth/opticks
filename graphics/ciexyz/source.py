#!/usr/bin/env python
"""
Compare simulated photon wavelengths against blackbody expectation

* good agreement from 150 to 820 nm
* simulated distrib is flat below 150nm ?  

  * are comparing final photon wavelengths, so maybe material props coming into play ... 

"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from env.numerics.npy.ana import Evt, Selection
from env.graphics.ciexyz.planck import planck

np.set_printoptions(suppress=True, precision=3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plt.ion()

    evt = Evt(tag="1", det="prism")

    sel = Selection(evt)

    w = evt.wavelength
    #w = sel.recwavelength(0)  

    wd = np.arange(60,810,20)

    mid = (wd[:-1]+wd[1:])/2.     # bin middle

    pl = planck(mid, 6500.)
    pl /= pl.sum()

    counts, edges = np.histogram(w, bins=wd )
    fcounts = counts.astype(np.float32)
    fcounts  /= fcounts.sum()

    plt.plot( edges[:-1], fcounts, drawstyle="steps-mid")

    plt.plot( mid,  pl ) 
    
    plt.axis( [w.min() - 100, w.max() + 100, 0, fcounts.max()*1.1 ]) 



