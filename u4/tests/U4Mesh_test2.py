#!/usr/bin/env python
"""
U4Mesh_test2.py
================

"""
import numpy as np, pyvista as pv
from opticks.ana.fold import Fold
SIZE = np.array([1280, 720])

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    label = "U4Mesh_test2.py GEOM %s : %s " % (f.NPFold_meta.GEOM[0], f.NPFold_meta.desc[0])
    pd = pv.PolyData(f.vtx, f.fpd)

    pl = pv.Plotter(window_size=SIZE*2)
    pl.add_text("%s" % label, position="upper_left")
    pl.add_mesh(pd, opacity=1.0, show_edges=True, lighting=True )
    pl.show_grid()
    pl.show()
pass


