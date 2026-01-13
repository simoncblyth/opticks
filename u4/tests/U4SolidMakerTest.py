#!/usr/bin/env python
"""
U4SolidMakerTest.py
====================

"""
import numpy as np, pyvista as pv
from opticks.ana.fold import Fold
from opticks.ana.pvplt import pvplt_show, pvplt_viewpoint
SIZE = np.array([1280, 720])

if __name__ == '__main__':
    f = Fold.Load("$FOLD/$SOLID",symbol="f")
    print(repr(f))

    label = "U4SolidMakerTest.py SOLID %s : %s " % (f.NPFold_meta.SOLID, f.NPFold_meta.desc)
    pd = pv.PolyData(f.vtx, f.fpd)

    pl = pv.Plotter(window_size=SIZE*2)
    pvplt_viewpoint(pl)

    pl.add_text("%s" % label, position="upper_left")
    pl.add_mesh(pd, opacity=1.0, show_edges=True, lighting=True )
    pl.show_grid()

    pvplt_show(pl)
pass


