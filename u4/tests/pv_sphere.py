#!/usr/bin/env python

#from np.fold import Fold
import numpy as np
import pyvista as pv
SIZE = np.array([1280, 720])
from pyvista.utilities.geometric_objects import Sphere                                                                               

if __name__ == '__main__':
    sp = Sphere(start_theta=0, end_theta=90, start_phi=10, end_phi=20 )  
    pl = pv.Plotter(window_size=SIZE*2)
    pl.add_text("pv_sphere.py", position="upper_left")
    pl.add_mesh(sp, opacity=1.0, show_edges=True, lighting=True )
    pl.show()
pass








