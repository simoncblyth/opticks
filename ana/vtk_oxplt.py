#!/usr/bin/env python
"""
vtk_oxplt.py
==============

Plot final photon positions 

This requires vtk, pyvista, ipython. 


Usage
------

Invoke from directory with ox.npy::

    cd /tmp/$USER/opticks/OKTest/evt/g4live/torch/1
    vtk_oxplt.py 
    ipython -i -- $(which vtk_oxplt.py)


OR give the directory as first argument::

    vtk_oxplt.py /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/


GUI
----

shift+drag 
    pan
two-finger-slide
    zoom 

"""
import logging 
log = logging.getLogger(__name__)
import sys
import numpy as np, os
import pyvista as pv   

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        os.chdir(sys.argv[1])
        log.info("chdir %s " % os.getcwd())
    pass
    np.set_printoptions(suppress=True, linewidth=200)
    ox = np.load("ox.npy")
    pl = pv.Plotter()
    pl.add_points(ox[:,0,:3] )
    log.info("Showing the VTK/pyvista plotter window, it may be hidden behind other windows. Enter Q to quit.")
    pl.show()


