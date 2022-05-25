#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold
import pyvista as pv 

if __name__ == '__main__':
    t = Fold.Load()

    print(t)

    gs = t.genstep
    p = t.photon

    pos = p[:,0,:3] 


    sphere = pv.Sphere(radius=17800)

    pl = pv.Plotter(window_size=2*np.array([1280, 720]))

    pl.add_points(pos) 
    pl.add_mesh(sphere, color="white", show_edges=True, style="wireframe")

    pl.show_grid()
    cp = pl.show()           



