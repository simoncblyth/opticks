#!/usr/bin/env python
"""
U4Mesh_test2_cf.py
====================

"""
import numpy as np, pyvista as pv
from opticks.ana.fold import Fold
SIZE = np.array([1280, 720])

if __name__ == '__main__':
    a = Fold.Load("$AFOLD",symbol="a")
    a_color = "red"
    a_label = "A : %s : %s : %s" % (a_color, a.NPFold_meta.GEOM[0], a.NPFold_meta.desc[0])
    print(repr(a))
    

    b = Fold.Load("$BFOLD",symbol="b")
    b_color = "blue"
    b_label = "B : %s : %s : %s " % (b_color, b.NPFold_meta.GEOM[0], b.NPFold_meta.desc[0])
    print(repr(b))


    a_offset = np.array([0,0,20], dtype=np.float64)
    b_offset = np.array([0,0, 0], dtype=np.float64)

    a_pd = pv.PolyData(a.vtx + a_offset, a.fpd)
    b_pd = pv.PolyData(b.vtx + b_offset, b.fpd)


    label = "\n".join(["U4Mesh_test2_cf.py", a_label, b_label ])
    print(label)

    pl = pv.Plotter(window_size=SIZE*2)
    pl.add_text("%s" % label, position="upper_left")
    pl.add_mesh(a_pd, opacity=1.0, show_edges=True, lighting=True, color=a_color )
    pl.add_mesh(b_pd, opacity=1.0, show_edges=True, lighting=True, color=b_color )

    pl.show_grid()
    pl.show()
pass


