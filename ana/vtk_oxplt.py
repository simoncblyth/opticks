#!/usr/bin/env python
"""
vtk_oxplt.py
==============

Plot final photon positions 

This requires vtk, pyvista, ipython. 


* https://docs.pyvista.org/plotting/plotting.html


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

    name = "ox"
    #name = "ht"

    a = np.load("%s.npy" % name)
    size = 2*np.array([1024,768], dtype=np.int32)
    pl = pv.Plotter(window_size=size)
    pl.add_points(a[:,0,:3] )
    pl.show_grid()
    log.info("Showing the VTK/pyvista plotter window, it may be hidden behind other windows. Enter q to quit.")
    cpos = pl.show()
    log.info(cpos)




