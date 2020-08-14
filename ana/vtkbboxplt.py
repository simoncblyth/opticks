#!/usr/bin/env python
"""
vtkbboxplt.py
==============

This requires vtk and pyvista. 

Use an ipython which has those by for example::

   ~/miniconda3/bin/ipython ~/opticks/ana/vtkbboxplt.py

OR enable that in ~/.bash_profile with::

  source $HOME/.miniconda3_setup  # py37+pyvista+vtk 


GUI
----

shift+drag 
    pan
two-finger-slide
    zoom 

"""
import numpy as np, os
import pyvista as pv   # requires vtk too, install with miniconda3 

from opticks.ana.cube import make_cube
from opticks.ana.ggeo import GGeo

if __name__ == '__main__':

    gg = GGeo()
    bbox = gg.bbox
    identity = gg.identity

    #s_bbox = bbox[identity[:,1] == 16]
    #s_bbox = bbox[:197]
    #s_bbox = bbox[197:]
    s_bbox = bbox 

    pl = pv.Plotter()
    for i in range(len(s_bbox)):
        bb = s_bbox[i]
        v,f,i = make_cube(bb)
        surf = pv.PolyData(v, i)
        pl.add_mesh(surf, opacity=0.5, color=True)
    pass
    pl.show()

        
