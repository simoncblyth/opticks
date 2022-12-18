#!/usr/bin/env python
"""
sboundary_test.py
===================

Start from qudarap/tests/qsim_test.py

Explain those two, at Brewster angle::

    In [5]: np.c_[pp[:,0,2],pp[:,1,2]]
    Out[5]: 
    array([[-0.555,  0.   , -0.832,  0.   , -0.555, -0.   ,  0.832,  0.   ],
           [-0.552,  0.098, -0.828,  0.   , -0.   , -1.   ,  0.   ,  0.   ],
           [-0.544,  0.195, -0.816,  0.   , -0.   , -1.   ,  0.   ,  0.   ],
           ..
           [ 0.   ,  1.   ,  0.   ,  0.   , -0.   , -1.   ,  0.   ,  0.   ],
           [ 0.054,  0.995,  0.082,  0.   , -0.   , -1.   ,  0.   ,  0.   ],
           [ 0.552,  0.098,  0.828,  0.   , -0.   , -1.   ,  0.   ,  0.   ],
           [ 0.555, -0.   ,  0.832,  0.   , -0.483,  0.491,  0.725,  0.   ],
           [ 0.552, -0.098,  0.828,  0.   , -0.   ,  1.   ,  0.   ,  0.   ],
           [ 0.512, -0.383,  0.769,  0.   , -0.   ,  1.   ,  0.   ,  0.   ],


Those are from initial pure P polarization that are forced to reflect 
which means that even when the coeff for reflection is zero for a particulr 
polarization are forced to calculate an outgoing polarization for it.  

What happens in that situation with the calculation is that a 
near zero length E2_r (S,P) vector gets dignified by normalization
into a vector with highly imprecise values that gets used
to construct the new pol::

              E2_r          (-0.000,-0.000)        length(E2_r)     0.0000
                 RR          (-0.491,-0.871)          length(RR)     1.0000


This is not a problem with the calc, just with forcing the reflect/transmit. 


To avoid this messing up the illustration switched to scaling the the length 
of the polarization vector with the relevant coefficient, 
so these polarizations that would be quashed dissappear into the origin


"""

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
import pyvista as pv
COLORS = "cyan red green blue cyan magenta yellow pink orange purple lightgreen".split()

import matplotlib as mp
CMAP = os.environ.get("CMAP", "hsv")  # Reds
cmap = mp.cm.get_cmap(CMAP)  


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    pp = t.pp

    polscale = float(os.environ.get("POLSCALE","1"))
    label = os.environ.get("TOPLINE", "sboundary_test.py")
    B = int(os.environ.get("B","1")) 

    pl = pvplt_plotter(label=label)   

    lim = slice(None)

    mom0 = pp[:,0,1,:3]
    pol0 = pp[:,0,2,:3]
    mom00 = mom0[0]     # all mom0 expected to be the same, just varying pol

    mom1 = pp[:,B,1,:3]
    pol1 = pp[:,B,2,:3]
    mom10 = mom1[0]     # all mom1 expected to be the same, just varying pol 

    pp[:,0,0,:3] = -mom0    # illustrative choice incident position on unit hemisphere
    pp[:,B,0,:3] =  mom1    # illustrative choice reflected/transmitted position on unit hemisphere

    print(" mom1.shape %s " % (str(mom1.shape)))  #  (16, 3) 

    ppv = np.c_[pp[:,0,2],pp[:,B,2]]  
    print("ppv = np.c_[pp[:,0,2],pp[:,B,2]] \n", repr(ppv))

    pvplt_viewpoint( pl ) 

    N = len(pp)
    ii = list(range(N))  

    wscale = False 
    wcut = True 

    for i in ii:
        frac = float(i)/float(N)
        polcol = cmap(frac)
        #polcol = COLORS[ i % len(COLORS)]
        pvplt_photon( pl, pp[i:i+1,0], polcol=polcol, polscale=polscale, wscale=wscale, wcut=wcut )
        pvplt_photon( pl, pp[i:i+1,B], polcol=polcol, polscale=polscale, wscale=wscale, wcut=wcut )
    pass


    pos = np.array( [[0,0,0]] , dtype=np.float32 )
    rvec = np.array( [[mom00[0],mom00[1],-mom00[2] ]], dtype=np.float32 ) 
    vec1 = np.array( [[mom10]], dtype=np.float32 ) 

    pvplt_lines( pl, pos, rvec )
    pvplt_lines( pl, pos, vec1 )

    lhs  = np.array( [[-1,0,0]], dtype=np.float32 )
    rhs  = np.array( [[ 1,0,0]], dtype=np.float32 )
    pvplt_add_line_a2b( pl, lhs, rhs )

    cp = pl.show() 
    
    print(cp)
    

