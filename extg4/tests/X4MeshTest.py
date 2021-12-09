#!/usr/bin/env python
"""
X4MeshTest.py
=============

Usage::

    x4                     # cd ~/opticks/extg4   
    x4 ; ./X4MeshTest.sh 

"""
import logging, numpy 
from opticks.ana.fold import Fold

try:
    import pyvista as pv
    from pyvista.plotting.colors import hexcolors
    #theme = "default"
    #theme = "dark"
    #theme = "paraview"
    theme = "document"
    pv.set_plot_theme(theme)
except ImportError:
    pv = None
    hexcolors = None
pass

class X4Mesh(object):
    @classmethod
    def MakePolyData(cls, fold):
        """
        * https://docs.pyvista.org/examples/00-load/create-poly.html
        """
        tri = fold.tri.reshape(-1,3)
        faces = np.zeros( (len(tri), 4 ), dtype=np.uint32 )
        faces[:,0] = 3   # all tri as quads were split within X4Mesh.cc  
        faces[:,1:] = tri 
        surf = pv.PolyData(fold.vtx, faces)
        return surf 

    @classmethod 
    def Load(cls, path):
        fold = Fold.Load(path)
        surf = cls.MakePolyData(fold)
        return cls(fold, surf)

    def __init__(self, fold, surf):
        self.fold = fold
        self.surf = surf



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    mesh = X4Mesh.Load("/tmp/X4MeshTest")
    mesh.surf.plot(cpos=[-1,-1,0.5])

  




