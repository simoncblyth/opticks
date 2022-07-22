#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *

TEST = os.environ["TEST"]
OPTS = os.environ["OPTS"] 

def test_AroundCylinder(t):
    """
    With OPTS 'TR,tr,R,T,r,t'

        np.all( a0 == a[:,0] )  
        np.all( a1 == a[:,1] )  

    """
    vopt = OPTS.split(",")
    
    assert vopt[0] == "TR" 
    assert vopt[1] == "tr" 

    a = t.AroundCylinder
    a0t = a[:,0].transpose(0,2,1)
    a1t = a[:,1].transpose(0,2,1)

    trs = a1t    ## arrows pointing inwards
    #trs = a0t   ## surpise "starfish"

    radius = 5. 
    O = np.array( [0,0,0] )
    Z = np.array( [0,0,1] )

    arrows = []
    for i in range(len(trs)):
        arr = pv.Arrow(direction=Z)
        arr.transform(trs[i])
        arrows.append(arr)
    pass
    cyl = pv.Cylinder(direction=Z, center=O, radius=radius ) 

    pl = pvplt_plotter()
    pl.add_mesh(cyl, style="wireframe")
    for arr in arrows:
        pl.add_mesh(arr)
    pass
    pl.show()




def test_AroundSphere(t):
    a = t.AroundSphere 

    a0t = a[:,0].transpose(0,2,1)
    a1t = a[:,1].transpose(0,2,1)

    trs = a1t   ## arrows pointing radially inwards from sphere surface
    #trs = a0t    ## arrows pointing out from S-pole

    radius = 5. 

    O = np.array( [0,0,0] )
    Z = np.array( [0,0,1] )

    arrows = []
    for i in range(len(trs)):
        arr = pv.Arrow(direction=Z)
        arr.transform(trs[i])
        arrows.append(arr)
    pass
    sph = pv.Sphere(center=O, radius=radius ) 

    pl = pvplt_plotter()
    pl.add_mesh(sph, style="wireframe")
    for arr in arrows:
        pl.add_mesh(arr)
    pass
    pl.show()






if __name__ == '__main__':
    t = Fold.Load()
    print(t)
    print("TEST:%s" % TEST)

    if TEST == "AroundCylinder":
        test_AroundCylinder(t)
    elif TEST == "AroundSphere":
        test_AroundSphere(t)
    pass


