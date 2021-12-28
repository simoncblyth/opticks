#!/usr/bin/env python

import numpy as np
import pyvista as pv


if __name__ == '__main__':

    path = os.path.expandvars(os.path.join("/tmp/$USER/opticks/CSGTargetGlobalTest/$MOI", "ce.npy")) 
    ce = np.load(path)
    
    pl = pv.Plotter(window_size=2*np.array([1280, 720]))
    pl.add_points(ce[:,:3])
    cp = pl.show()                                   

