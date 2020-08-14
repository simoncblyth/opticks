#!/usr/bin/env python
"""
CAUTION: this approach to 3d plotting is horribly slow
and should only be used for very small geometries.  For
much better performance::

    grep pyvista *.py 

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def polyplot(ax, verts, faces):
    pcoll = Poly3DCollection(faces, linewidths=1, edgecolors='k')
    pcoll.set_facecolor((0,0,1,0.1))
    ax.add_collection3d(pcoll)
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], s=0)
    #ax.set_aspect('equal')  # fails in py3 (anaconda3)
pass


