#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *

TEST = os.environ["TEST"]

def test_AroundCylinder(t):
    a0 = t.AroundCylinder0 
    a1 = t.AroundCylinder1  
    ## both these have the translate in the last row with the rotation transposed between a0 and a1

    a0t = a0.transpose(0,2,1)
    a1t = a1.transpose(0,2,1)
  
    # NOPE: not when both rotate and translate 
    #assert np.all( a0t == a1 )   # leave top dimension as is, just transpose the 4x4 matrix dimensions 
    #assert np.all( a1t == a0 )  

    trs = a1t  ## arrows pointing outwards   ## why not inwards ?
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
    a0 = t.AroundSphere0 
    a1 = t.AroundSphere1  # with flip

    a0t = a0.transpose(0,2,1)
    a1t = a1.transpose(0,2,1)
 
    #assert np.all( a0.transpose(0,2,1) == a1 )   # leave top dimension as is, just transpose the 4x4 matrix dimensions 
    #assert np.all( a1.transpose(0,2,1) == a0 )  

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


