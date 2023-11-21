#!/usr/bin/env python
"""
G4CXTest_GEOM.py
======================

"""
import os, logging, textwrap, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt, SAB
from opticks.ana.nbase import chi2_pvalue

MODE = int(os.environ.get("MODE","3"))
PICK = os.environ.get("PICK","A")

if MODE in [2,3]:
    try:
        import pyvista as pv
        from opticks.ana.pvplt import pvplt_plotter, pvplt_viewpoint
    except ImportError:
        pv = None
    pass
else:
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
    ee = {'A':a, 'B':b} 

    print("[--- ab = SAB(a,b) ----")
    ab = None if a is None or b is None else SAB(a,b) 
    print("]--- ab = SAB(a,b) ----")

    print("[----- repr(ab) ")
    print(repr(ab))
    print("]----- repr(ab) ")


    EXPR_ = r"""
    np.c_[np.unique(a.q, return_counts=True)] 
    np.c_[np.unique(b.q, return_counts=True)] 
    """
    EXPR = list(filter(None,textwrap.dedent(EXPR_).split("\n")))
    for expr in EXPR:eprint(expr, locals(), globals() )

    context = "PICK=%s MODE=%d  ~/opticks/g4cx/tests/G4CXTest_GEOM.sh " % (PICK, MODE )
    print(context)

    for Q in PICK:
        e = ee.get(Q,None)
        if e is None:continue

        elabel = "%s : %s " % ( e.symbol.upper(), e.f.base )
        label = context + " ## " + elabel

        if hasattr(e.f, 'photon'): 
            pos = e.f.photon[:,0,:3]
        else:
            pos = None
        pass

        if hasattr(e.f, 'hit'): 
            hit = e.f.hit[:,0,:3]
        else:
            hit = None
        pass

        if hasattr(e.f, 'record'):
            sel = np.where(e.f.record[:,:,2,3] > 0) # select on wavelength to avoid unfilled zeros
            poi = e.f.record[:,:,0,:3][sel]
        else:
            poi = None
        pass

        if MODE == 3 and not pv is None:
            pl = pvplt_plotter(label)
            pvplt_viewpoint(pl) # sensitive EYE, LOOK, UP, ZOOM envvars eg EYE=0,-3,0 

            if not poi is None:
                pl.add_points( poi, color="green", point_size=3.0 )
            pass
            if not pos is None:
                pl.add_points( pos, color="red", point_size=3.0 )
            pass
            if not hit is None:
                pl.add_points( hit, color="cyan", point_size=3.0 )
            pass

            pl.show_grid()
            cp = pl.show()
        pass
    pass
        

