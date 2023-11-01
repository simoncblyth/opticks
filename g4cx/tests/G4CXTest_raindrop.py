#!/usr/bin/env python

import os, logging, textwrap, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt, SAB

try:
    import pyvista as pv
    from opticks.ana.pvplt import pvplt_viewpoint
except ImportError:
    pv = None
pass

def eprint(expr, l, g ):
    print(expr)
    try:
       val = eval(expr,l,g)
    except AttributeError:
       val = "eprint:AttributeError"
    pass
    print(val)
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
        if e is None:continue
        pos = e.f.photon[:,0,:3]
        sel = np.where(e.f.record[:,:,2,3] > 0) # select on wavelength to avoid unfilled zeros
        poi = e.f.record[:,:,0,:3][sel]

        if not pv is None:
            size = np.array([1280, 720])
            pl = pv.Plotter(window_size=size*2 )
            pvplt_viewpoint(pl) # sensitive EYE, LOOK, UP, ZOOM envvars eg EYE=0,-3,0 
            pl.add_points( poi, color="green", point_size=1.0 )
            pl.add_points( pos, color="red", point_size=1.0 )
            pl.show_grid()
            cp = pl.show()
        pass
    pass
        

