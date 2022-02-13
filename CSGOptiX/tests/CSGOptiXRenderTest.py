#!/usr/bin/env python

"""

    epsilon:CSGOptiX blyth$ i tests/CSGOptiXRenderTest.py 
       a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
       b :         (3, 3, 4, 4) :  select the central portion of the image array  


Note that the first dimension is the smaller vertical 1080 one (in landscape aspect). 
Selecting vertical strip 3 pixels wide and 21 pixels high in the middle of the frame::

    epsilon:CSGOptiX blyth$ DYDX=10,1 i tests/CSGOptiXRenderTest.py 
       a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
       b :        (21, 3, 4, 4) :  select the central portion of the image array  


Selecting 3x3 square in middle, happens to make an "H"::

    DYDX=1,1 i tests/CSGOptiXRenderTest.py


    DYDX=4,4 i tests/CSGOptiXRenderTest.py


"""

import numpy as np
import platform
import matplotlib.pyplot as plt                   
from opticks.ana.eget import efloat_, efloatlist_, eint_, eintlist_

def isectpath():
    geochain_suffix = "_Darwin" if platform.system() == "Darwin" else ""  
    geom = os.environ.get("GEOM", "GeneralSphereDEV")
    cvdver = os.environ.get("CVDVER", "cvd0/50001" ) 
    fmt = "/tmp/$USER/opticks/GeoChain%(geochain_suffix)s/%(geom)s/CSGOptiXRenderTest/%(cvdver)s/ALL/top_i0_/cxr_geochain_%(geom)s_ALL_isect.npy" % locals() 
    path = os.path.expandvars(fmt)
    return path 

def outpath_(dy, dx):
    outdir = os.path.expandvars("/tmp/$USER/opticks/CSGOptiX/CSGOptiXRenderTest")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    pass
    outname = "dy_%d_dx_%d.npy" % ( dy, dx ) 
    outpath = os.path.join(outdir, outname)
    return outpath 


if __name__ == '__main__':

    path = isectpath()
    a = np.load(path)

    fmt = " %3s : %20s : %s "
    print(fmt % ( "a", str(a.shape), path ))

    ny, nx, ni, nj = a.shape
    assert ni == 4
    assert nj == 4

    dy,dx = eintlist_("DYDX", "1,1")

    mid_y, mid_x = ny//2, nx//2
    ys = slice(mid_y-dy, mid_y+dy+1) 
    xs = slice(mid_x-dx, mid_x+dx+1) 

    b = a[ys,xs] 

    outpath = outpath_(dy, dx)
    np.save(outpath, b )
    print(fmt % ( "b", str(b.shape), outpath ))
    
    a_result = a[:,:,0,:3] 
    b_result = b[:,:,0,:3] 

    fig, axs = plt.subplots(2)            
    axs[0].imshow( a_result ) 
    axs[1].imshow( b_result )
    fig.show()









