#!/usr/bin/env python
"""
vtk_oxplt.py
==============

Plot final photon positions 

This requires vtk, pyvista, ipython. 


Usage
------

::

    cd /tmp/$USER/opticks/OKTest/evt/g4live/torch/1

    vtk_oxplt.py 

    ipython -i -- $(which vtk_oxplt.py)


GUI
----

shift+drag 
    pan
two-finger-slide
    zoom 

"""
import numpy as np, os
import pyvista as pv   

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=200)
    ox = np.load("ox.npy")
    pl = pv.Plotter()
    pl.add_points(ox[:,0,:3] )
    pl.show()


