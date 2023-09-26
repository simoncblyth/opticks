#!/usr/bin/env python
from np.fold import Fold
import numpy as np
import pyvista as pv
SIZE = np.array([1280, 720])

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))
    pd = pv.PolyData(f.vtx, f.fpd)   # tri and/or quad
    pl = pv.Plotter(window_size=SIZE*2)
    pl.add_text("U4Mesh_test.sh", position="upper_left")
    pl.add_mesh(pd, opacity=1.0, show_edges=True, lighting=True )
    pl.show()
pass


