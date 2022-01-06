#!/usr/bin/env python
"""

"""
import numpy as np
import pyvista as pv


if __name__ == '__main__':

    path = os.path.expandvars(os.path.join("/tmp/$USER/opticks/CSGTargetGlobalTest/$MOI", "ce.npy")) 
    ce = np.load(path)

    print("path %s " % path )
    print("ce.shape %s " % str(ce.shape))

    r = np.sqrt(np.sum(ce[:,:3]*ce[:,:3], axis=1))                                                                                                                                                    
    rmin = r.min()
    rmax = r.max()
    print("rmin %s rmax %s rmax-rmin %s " % (rmin, rmax, rmax-rmin))

    pl = pv.Plotter(window_size=2*np.array([1280, 720]))
    pl.add_points(ce[:,:3])

    pl.show_grid()
    cp = pl.show()                                   




