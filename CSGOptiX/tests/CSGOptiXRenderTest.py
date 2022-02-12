#!/usr/bin/env python

import numpy as np
import platform
import matplotlib.pyplot as plt                   

def isectpath():
    geochain_suffix = "_Darwin" if platform.system() == "Darwin" else ""  
    geom = os.environ.get("GEOM", "GeneralSphereDEV")
    cvdver = os.environ.get("CVDVER", "cvd0/50001" ) 
    fmt = "/tmp/$USER/opticks/GeoChain%(geochain_suffix)s/%(geom)s/CSGOptiXRenderTest/%(cvdver)s/ALL/top_i0_/cxr_geochain_%(geom)s_ALL_isect.npy" % locals() 
    path = os.path.expandvars(fmt)
    return path 

if __name__ == '__main__':

    path = isectpath()
    a = np.load(path)

    fmt = " %3s : %20s : %s "
    print(fmt % ( "a", str(a.shape), path ))

    nx, ny, ni, nj = a.shape
    assert ni == 4
    assert nj == 4

    dx,dy = 1,1

    mid_x, mid_y = nx//2, ny//2
    xs = slice(mid_x-dx, mid_x+dx+1) 
    ys = slice(mid_y-dy, mid_y+dy+1) 


    b = a[xs,ys] 
    print(fmt % ( "b", str(b.shape), " select the central portion of the image array " ))
    
    a_result = a[:,:,0,:3] 
    b_result = b[:,:,0,:3] 

    fig, axs = plt.subplots(2)            
    axs[0].imshow( a_result ) 
    axs[1].imshow( b_result )
    fig.show()


