#!/usr/bin/env python
"""
SEventTest.py
=================

::

    ~/o/sysrap/tests/SEventTest.sh pdb 

"""

import os, numpy as np
from opticks.ana.fold import Fold

try:
    import pyvista as pv
except ImportError:
    pv = None
pass



def test_transform():
    dtype = np.float32
    p0 = np.array([1,2,3,1], dtype)
    p1 = np.zeros( (len(cegs), 4), dtype ) 
    p1x = cegs[:,1]

    # transform p0 by each of the genstep transforms 
    for i in range(len(cegs)): p1[i] = np.dot( p0, cegs[i,2:] )
    pos = p1[:,:3]   
    return pos 

def test_ppa(t):
    pos = t.ppa[:,0,:3]
    dir = t.ppa[:,1,:3]
    #pv = None

    if not pv is None:
        size = np.array([1280, 720])
        pl = pv.Plotter(window_size=size*2 ) 
        pl.add_points( pos, color='#FF0000', point_size=10.0 ) 
        pl.add_arrows( pos, dir, mag=50, color='#00FF00', point_size=1.0 ) 
        pl.show_grid()
        cp = pl.show()    
    pass



if __name__ == '__main__':
    TEST = os.environ["TEST"]
    print("TEST:%s" % TEST) 
    t = Fold.Load("$FOLD", symbol="t")
    print(t)

    if TEST == "MakeTorchGenstep":
        a = t.torch
        print("a = t.torch\n",a)
        print("a.shape\n", a.shape) 
        e = "a[:,0].view(np.int32)"
        print(e,"\n",eval(e))
    pass
   

 


