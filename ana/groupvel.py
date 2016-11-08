#!/usr/bin/env python
"""

G4 uses funny raster, to avoid dropping a bin::

    ///     hmmm a difference of bins is needed, but in order not to loose a bin
    ///     a tricky manoever is used of using the 1st and last bin and 
    ///     the average of the body bins
    ///     which means the first bin is half width, and last is 1.5 width
    ///
    ///         0  +  1  +  2  +  3  +  4  +  5        <--- 6 original values
    ///         |    /     /     /     /      |
    ///         |   /     /     /     /       |
    ///         0  1     2     3     4        5        <--- still 6 
    ///


"""

import os, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.proplib import PropLib

if __name__ == '__main__':
    ok = opticks_main()
    mlib = PropLib("GMaterialLib")
    gdls = mlib("GdDopedLS")

    g4 = np.load(os.path.expandvars("$TMP/gdls.npy"))    # written by cfg4/tests/GROUPVELTest.cc


    ri = gdls[0,:,0]
    wl = np.linspace(60,820,39)
    c_light = 299.792
    assert ri.shape == wl.shape and len(ri) == len(wl)

    n0,n1= ri[:-1],ri[1:]
    w0,w1 = wl[:-1],wl[1:]

    
    nn = np.zeros_like(ri)

    nn[0] = ri[0]
    nn[1:-1] = (ri[1:-1] + ri[:-2])/2
    nn[-1] = ri[-1]

    ds = np.zeros_like(ri)
    ds[1:] = (n1-n0)/np.log(w1/w0)    # tricky double flip ?, from reciprocal energy vs wl and reversed ordering 
    ds[0] = ds[1]

    vg = c_light/(nn + ds)
    vg0 = c_light/nn

    msk = np.logical_or( vg < 0, vg > vg0)

    vgm = vg.copy()
    vgm[msk] = vg0[msk]


    #print "".join(map(lambda _:"%10s"%_, "n0,n1,w0,w1,ds,vg0,vg,msk,vgm".split(",")))
    print np.dstack([nn,ds,vg0,vg,msk,vgm])







