#!/usr/bin/env python
from np.fold import Fold
import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

SIZE = np.array([1280, 720])

if __name__ == '__main__':
    f = Fold.Load("$FOLD/$SOLID", symbol="f")
    print(repr(f))
    title = os.environ.get("TITLE", "U4Mesh_test.sh")

    if pv == None: 
        print("SKIP plotting as no pyvista")
    else:    
        #pd = pv.PolyData(f.vtx, f.fpd)   # tri and/or quad
        pd = pv.PolyData(f.vtx, f.tpd)   # tri and/or quad

        pl = pv.Plotter(window_size=SIZE*2)
        pl.add_text(title, position="upper_left")
        pl.add_mesh(pd, opacity=1.0, show_edges=True, lighting=True )
        pl.show()
    pass
pass


