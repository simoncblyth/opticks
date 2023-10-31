#!/usr/bin/env python

import os, logging, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt, SAB

try:
    import pyvista as pv
except ImportError:
    pv = None
pass



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    a = SEvt.Load("$AFOLD", symbol="a")
    print(repr(a))
    b = SEvt.Load("$BFOLD", symbol="b")
    print(repr(b))

    expr = "np.c_[np.unique(b.q, return_counts=True)]"
    print(expr)
    print(eval(expr))

    pos = b.f.photon[:,0,:3]
    poi = b.f.record[:,:,:,:3].reshape(-1,3)

    if not pv is None:
        size = np.array([1280, 720])
        pl = pv.Plotter(window_size=size*2 )
        pl.add_points( poi, color="green", point_size=10.0 )
        pl.add_points( pos, color="red", point_size=10.0 )
        pl.show_grid()
        cp = pl.show()
    pass
    



