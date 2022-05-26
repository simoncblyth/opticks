#!/usr/bin/env python
"""






"""
import numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import * 
from opticks.sysrap.sframe import sframe 

import pyvista as pv 

if __name__ == '__main__':
    t = Fold.Load()

    print(t)

    gs = t.genstep
    p = t.photon
    frame = sframe.Load(t.base)

    print(frame)


    pos = p[:,0,:3] 
    mom = p[:,1,:3] 
    pol = p[:,2,:3] 



    pl = pv.Plotter(window_size=2*np.array([1280, 720]))

    pl.add_points( pos, color="magenta", point_size=1.0 )

    #sphere = pv.Sphere(radius=17800)
    #pl.add_mesh(sphere, color="white", show_edges=True, style="wireframe")

    # have to diddle mag to get something visible depending 
    # on the size of the region that is plotted

    pl.add_arrows( pos, mom, mag=50, show_scalar_bar=False, color="red" )

    #pvplt_arrows( pl, pos, mom, factor=1000 )    
    #pvplt_polarized( pl, pos, mom, pol, factor=2000 )    


    pl.show_grid()
    cp = pl.show()           



