#!/usr/bin/env python

import os, logging, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.gridspec import GridSpec, X, Y, Z
from opticks.ana.npmeta import NPMeta

SIZE = np.array([1280, 720])

log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

try:
    import matplotlib.pyplot as mp  
except ImportError:
    mp = None
pass
#mp = None

try:
    import pyvista as pv
    from pyvista.plotting.colors import hexcolors  
    themes = ["default", "dark", "paraview", "document" ]
    pv.set_plot_theme(themes[1])
except ImportError:
    pv = None
    hexcolors = None
pass
#pv = None





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    path = os.path.expandvars("$CFBASE/CSGIntersectSolidTest/$GEOM") 
    fold = Fold.Load(path) 


    pl = pv.Plotter(window_size=SIZE*2 ) 

    pos = fold.isect[:,1,:3]
    dir_ = fold.isect[:,0,:3]

    pl.add_points( pos, color="white" )
    #pl.add_arrows( pos, dir_, color="white", mag=10 )
    cp = pl.show()



if 0:
    pos = fold.photons[:,0,:3]
    dir_ = fold.photons[:,1,:3]

    pl.add_arrows( pos, dir_, color="white", mag=10 )
    cp = pl.show()


