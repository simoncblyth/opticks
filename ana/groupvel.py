#!/usr/bin/env python
"""

G4 uses funny raster, to avoid dropping a bin::


"""

import os, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.proplib import PropLib

hc = 1239.8419     # eV.nm
c_light = 299.792  # mm/ns


def g4_bintrick(aa):
    """
    :param aa:
    :return bb:

    This curious bin shifting is used by G4 GROUPVEL calc

    In order not to loose a bin
    a tricky manoever of using the 1st and last bin and 
    the average of the body bins
    which means the first bin is half width, and last is 1.5 width
    
    ::

            0  +  1  +  2  +  3  +  4  +  5        <--- 6 original values
            |    /     /     /     /      |
            |   /     /     /     /       |
            0  1     2     3     4        5        <--- still 6 
  
    """
    bb = np.zeros_like(aa)
    bb[0] = aa[0]
    bb[1:-1] = (aa[1:-1] + aa[:-2])/2.
    bb[-1] = aa[-1]
    return bb


def np_gradient(y, edge_order=1):
    """
    ::

        Take a look at np.gradient??

        The gradient is computed using second order accurate central differences
        in the interior and either first differences or second order accurate 
        one-sides (forward or backwards) differences at the boundaries. The
        returned gradient hence has the same shape as the input array.

          0    <--- (0,1)  
  
          1    <--- (0,2)/2
          2    <--- (1,3)/2
          3    <--  (2,4)/2
          4    <--  (3,5)/2
          5    <--  (4,6)/2
          6    <--  (5,7)/2
          7    <--  (6,8)/2
          8    <--  (7,9)/2

          9    <--  (8,9) 


        In [10]: y = np.arange(10, dtype='f')


        In [14]: y[:-2]   # skip bins N-2,N-1  :  ie N-2 values
        Out[14]: array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.], dtype=float32)

        In [13]: y[2:]   # skip bins 0,1 : ie N-2 values
        Out[13]: array([ 2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.], dtype=float32)

        In [15]: y[2:] - y[:-2]    # between N-2 values -> N-2 values
        Out[15]: array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.], dtype=float32)

        In [16]: (y[2:] - y[:-2])/2.
        Out[16]: array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)

    """
    assert edge_order == 1
    out = np.zeros_like(y)
    out[0] = y[1] - y[0]
    out[1:-1] = (y[2:]-y[:-2])/2.0
    out[-1] = y[-1] - y[-2]
    return o



def g4style_edom(okmat,g4mat,wl):
    """
    Reproduce G4 GROUPVEL calc by migrating from wavelength to energy 
    domain and duplicating the tricky bin shift manoever

                c         
    vg =  ---------------        # angular freq proportional to E for light     
            n + E dn/dE

    G4 using this energy domain approach approximating the dispersion part E dn/dE as shown below

                c                  n1 - n0         n1 - n0               dn        dn    dE          
    vg =  -----------       ds = ------------  =  ------------     ~   ------  =  ---- ------- =  E dn/dE 
           nn +  ds               log(E1/E0)      log E1 - log E0      d(logE)     dE   dlogE        


    Start by convert to energy and flipping order of energy 
    and refractive values to correspond to increasing energy domain 
    """
    ri = okmat[0,:,0]  # refractive index values on wavelength domain
    en = hc/wl[::-1]   # eV    
    ri = ri[::-1] 

    assert ri.shape == en.shape and len(ri) == len(en)

    n0,n1= ri[:-1],ri[1:]
    e0,e1 = en[:-1],en[1:]

    nn = g4_bintrick(ri)
    ee = g4_bintrick(en)

    ds = np.zeros_like(ri)  # dispersion term
    ds[1:] = (n1-n0)/np.log(e1/e0)  ## have lost a bin from the diff ... 
    ds[0] = ds[1]                   ## duping into first value

    # ds is native to midbin anyhow

    vg0 = c_light/nn
    vg = c_light/(nn + ds)

    msk = np.logical_or( vg < 0, vg > vg0)
    vgm = vg.copy()
    vgm[msk] = vg0[msk]

    g4vg = g4mat[1,::-1,2]

    labels = "hc/en,en,hc/ee,hc/ee-hc/en,ee,nn,ds,vg0,vg,msk,vgm,g4vg,vgm-g4vg,vgi"

    vgi = np.interp( en, ee, vgm )   # linear interpolation onto original energy values

    print "".join(map(lambda _:"%11s"%_, labels.split(",")))
    print np.dstack([hc/en,en,hc/ee,hc/ee-hc/en,ee,nn,ds,vg0,vg,msk,vgm,g4vg,vgm-g4vg, vgi])




def gradient_edom(okmat, g4mat, wl):
    """
                c         
    vg =  ---------------        # angular freq proportional to E for light     
            n + E dn/dE

    """
    ri = okmat[0,:,0]  # refractive index values on wavelength domain
    en = hc/wl[::-1]   # eV    
    ri = ri[::-1] 

    dn = np.gradient(ri)  # keeps same number of bins
    de = np.gradient(en)

    vg0 = c_light/ri 
    vg = c_light /( ri + en*dn/de ) 

    msk = np.logical_or( vg < 0, vg > vg0)
    vgm = vg.copy()
    vgm[msk] = vg0[msk]

    g4ee = bintrick(en)
    g4vg = g4mat[1,::-1,2]
    g4vgi = np.interp(en,g4ee,g4vg)  # interpolate g4 results from tricky bins to original bins

    labels = "en,ri,hc/en,dn,de,vg0,vg,msk,vgm,g4ee,g4vg,g4vgi,vgm-g4vgi"
    print "".join(map(lambda _:"%11s"%_, labels.split(",")))
    print np.dstack([en,ri,hc/en,dn,de,vg0,vg,msk,vgm,g4ee,g4vg,g4vgi,vgm-g4vgi])
 



def g4style_wdom(okmat,g4mat,wl):
    """
    Attempt to stay in wavelength domain, and see how close can get to g4

    Not keen on giving groupvel values on a different domain to standard input domain, 
    but run with that for now whilst checking can do the calc in wavelength domain. 

     c          dn  
     -   +   c  ---
     n          dlogw


    Hmm seems not worth the effort to get wdom calc to match when 
    easy enough to migrate into edom first.
    """
    pass
    ri = okmat[0,:,0]  # refractive index values on wavelength domain
    n0,n1= ri[:-1],ri[1:]
    w0,w1 = wl[:-1],wl[1:]

    nn = bintrick(ri)
    ww = bintrick(wl,reciprocal=True)
    
    vg0 = c_light/nn 

    ds = np.zeros_like(ri)
    ds[1:] = (n1-n0)/np.log(w1/w0) 
    ds[0] = ds[1]

    vg = vg0 + c_light*ds

    msk = np.logical_or( vg < 0, vg > vg0)
    vgm = vg.copy()
    vgm[msk] = vg0[msk]

    g4vg = g4mat[1,:,2]

    labels = "ww,nn,ds,vg0,vg,msk,vgm,g4vg,vgm-g4vg"
    print "".join(map(lambda _:"%11s"%_, labels.split(",")))
    print np.dstack([ww,nn,ds,vg0,vg,msk,vgm,g4vg,vgm-g4vg])





if __name__ == '__main__':
    ok = opticks_main()
    mlib = PropLib("GMaterialLib")

    name = "GdDopedLS"
    g4name = "gdls"

    wl = np.linspace(60,820,39)   # standard wavelength domain with increasing wavelenth (nm)
    okmat = mlib(name)
    g4mat = np.load(os.path.expandvars("$TMP/%s.npy" % g4name))    # written by cfg4/tests/GROUPVELTest.cc

    g4style_edom(okmat,g4mat,wl)
    gradient_edom(okmat,g4mat,wl)



