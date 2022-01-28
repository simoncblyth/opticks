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

    dir_ = fold.isect[:,0,:3]
    t = fold.isect[:, 0, 3]
    pos = fold.isect[:,1,:3]
    sd = fold.isect[:,1,3]

    sd_cut = -1e-3
    spurious = sd < sd_cut 

    ray_origin = fold.isect[:, 2, :3]
    ray_direction = fold.isect[:, 3, :3]


    s_t = t[spurious]
    s_pos = pos[spurious]
    s_dir = dir_[spurious]
    s_isect = fold.isect[spurious]

    s_ray_origin = ray_origin[spurious]
    s_ray_direction = ray_direction[spurious]

    log.info( "sd_cut %10.4g sd.min %10.4g sd.max %10.4g num spurious %d " % (sd_cut, sd.min(), sd.max(), len(s_pos)))

    pl.add_points( pos, color="white" )
    #pl.add_arrows( pos, dir_, color="white", mag=10 )

    if len(s_pos) > 0:
        pl.add_points( s_pos, color="red" )
        pl.add_arrows( s_pos, s_dir, color="red", mag=10 )
        pl.add_arrows( s_ray_origin, s_ray_direction, color="blue", mag=s_t )
    else:
        log.info("no spurious with sd_cut %10.4g " % sd_cut)
    pass

    cp = pl.show()



if 0:
    pos = fold.photons[:,0,:3]
    dir_ = fold.photons[:,1,:3]

    pl.add_arrows( pos, dir_, color="white", mag=10 )
    cp = pl.show()


