#!/usr/bin/env python
"""
CSGSignedDistanceFieldTest.py
===============================

see env/graphics/pyvista_/sdContour.py

"""
import os 
import pyvista as pv 
import numpy as np
from opticks.ana.fold import Fold
from opticks.ana.npmeta import NPMeta

SIZE = np.array([1280, 720])

class CSGSignedDistanceFieldTest(object):
    BASE = "$TMP/CSG/CSGSignedDistanceFieldTest" ; 
    def __init__(self, geom):
        fold = Fold.Load( self.BASE, geom )
        self.geom = geom
        self.fold = fold
        self.sdf = fold.sdf 
        self.xyzd = fold.xyzd 



if __name__ == '__main__':
    geom = os.environ.get("GEOM", "UnionBoxSphere")
    t = CSGSignedDistanceFieldTest(geom)

    sdf = t.sdf
    xyzd = t.xyzd


    sdf_meta = t.fold.sdf_meta 
    pm = NPMeta(t.fold.sdf_meta)  

    ox = float(pm.find("ox:"))
    oy = float(pm.find("oy:"))
    oz = float(pm.find("oz:"))

    sx = float(pm.find("sx:"))
    sy = float(pm.find("sy:"))
    sz = float(pm.find("sz:"))

    assert len(sdf.shape) == 3, "unexpected sdf.shape %s " % str(sdf.shape)  
    ni, nj, nk = sdf.shape


    dims=(ni, nj, nk)
    spacing=(sx,sy,sz)
    origin=(ox,oy,oz)

    grid = pv.UniformGrid(dims, spacing, origin)
    print(grid)

    #x, y, z = grid.points.T

    values = sdf.ravel()
    values_2 = xyzd.reshape(-1,4)[:,3]
    assert np.all( values == values_2 )

    grid.point_arrays["values"] = values

    method = ['marching_cubes','contour','flying_edges'][1]
    isovalue = 0 
    num_isosurfaces = 1 
    mesh = grid.contour(num_isosurfaces, scalars="values", rng=[isovalue, isovalue], method=method )

    show_edges = "EDGES" in os.environ   
    pl = pv.Plotter(window_size=SIZE*2)
    pl.add_mesh(mesh, smooth_shading=False, color='tan', show_edges=show_edges  )   # style='wireframe' 
    pl.show_grid()
    pl.show()





