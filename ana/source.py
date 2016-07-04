#!/usr/bin/env python
"""
Wavelength Distribution Debugging
====================================

Compare simulated photon wavelengths against blackbody expectation

* still a hint of "ringing steps" from 200:400nm, but seems acceptable 
  (TODO: try increasing icdf bins from 1024 to identify) 


[ISSUE] wp last bin elevated
----------------------------- 

::

    In [69]: plt.close();plt.hist(wp, bins=200)

    ,  2215.,  2158.,  2046.,  2017.,  2052.,  2111.,  2565.]),



[FIXED] Bug with w0 sel.recwavelength(0)  
-----------------------------------------

Without selection sel.recwavelength(0) from ggv-newton:

* length of 500000

* three bin spike at lower bound around 60nm, comprising about 7000 photons
  (not present in the uncompressed wp)

  **FIXED WHEN AVOID WAVELENGTH DOMAIN DISCREPANCY BETWEEN SOURCES AND COMPRESSION**  

* plateau from 60~190 nm

  **MADE MUCH LESS OBJECTIONABLE BY INCREASING ICDF BINS FROM 256 TO 1024** 

* normal service resumes above 190nm with good
  match to Planck black body curve

* 256 unique linspaced values, a result of the compression:: 

    In [36]: np.allclose(np.linspace(60,820,256),np.unique(w))  # upper changed 810 to 820 by the fix
    Out[36]: True

"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from opticks.ana.ana import Evt, Selection
from env.graphics.ciexyz.planck import planck

np.set_printoptions(suppress=True, precision=3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plt.ion()

    evt = Evt(tag="1", det="rainbow")

    sel = Selection(evt)

    wp = evt.wavelength
    w0 = sel.recwavelength(0)  

    w = wp
    #w = w0

    wd = np.linspace(60,820,256) - 1.  
    # reduce bin edges by 1nm to avoid aliasing artifact in the histogram

    mid = (wd[:-1]+wd[1:])/2.     # bin middle

    pl = planck(mid, 6500.)
    pl /= pl.sum()

    counts, edges = np.histogram(w, bins=wd )
    fcounts = counts.astype(np.float32)
    fcounts  /= fcounts.sum()


    plt.close()

    plt.plot( edges[:-1], fcounts, drawstyle="steps-mid")

    plt.plot( mid,  pl ) 
    
    plt.axis( [w.min() - 100, w.max() + 100, 0, fcounts.max()*1.1 ]) 


    #plt.hist(w, bins=256)   # 256 is number of unique wavelengths (from record compression)



