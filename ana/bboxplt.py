#!/usr/bin/env python
"""
bboxplt.py
============

This uses the 3d capabilities of matplotlib.
For better performance try: vtkbboxplt.py

The slowness of this visualization even with a small number
(few hundred) of global volumes prompted me to try installing mayavi
and dependencies via miniconda3 with python37.
Mayavi did not work (giving blank OpenGL windows) 
but in the process I got vtk and pyvista working 
which provides fast and convenient 3d viz.


"""
import numpy as np, os
import matplotlib.pyplot as plt 

from opticks.ana.plt3d import polyplot
from opticks.ana.cube import make_cube
from opticks.ana.ggeo import GGeo

if __name__ == '__main__':

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    gc = GGeo()
    bbox = gc.bbox
    identity = gc.identity

    #s_bbox = bbox[identity[:,1] == 16]
    s_bbox = bbox[:197]


    for i in range(len(s_bbox)):
        bb = s_bbox[i]
        v,f,i = make_cube(bb)
        polyplot(ax, v, f)
    pass
    plt.show()
  
        
