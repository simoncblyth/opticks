#!/usr/bin/env python

import os, logging, textwrap, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt, SAB

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

def eprint(expr, l, g ):
    print(expr)
    print(eval(expr,l,g))
pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    a = SEvt.Load("$AFOLD", symbol="a")
    print(repr(a))
    b = SEvt.Load("$BFOLD", symbol="b")
    print(repr(b))

    EXPR_ = r"""
    np.c_[np.unique(a.q, return_counts=True)] 
    np.c_[np.unique(b.q, return_counts=True)] 
    """
    EXPR = list(filter(None,textwrap.dedent(EXPR_).split("\n")))
    for expr in EXPR:eprint(expr, locals(), globals() )

    for e in [a,b]:
        pos = e.f.photon[:,0,:3]
        poi = e.f.record[:,:,:,:3].reshape(-1,3)

        if not pv is None:
            size = np.array([1280, 720])
            pl = pv.Plotter(window_size=size*2 )
            pl.add_points( poi, color="green", point_size=10.0 )
            pl.add_points( pos, color="red", point_size=10.0 )
            pl.show_grid()
            cp = pl.show()
        pass
    pass
        

