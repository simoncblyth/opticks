#!/usr/bin/env python

import textwrap
import numpy as np, pyvista as pv
from opticks.ana.fold import Fold
SIZE = np.array([1280, 720])

if __name__ == '__main__':
    f = Fold.Load("$RECORD_FOLD",symbol="f")
    print(repr(f))

    p = f.record[:,:,0,:3].reshape(-1,3)
    t = f.record[:,:,0,3].reshape(-1)

    EXPR = r"""
    np.max(p,axis=0)
    np.min(p,axis=0)
    """
    for expr in list(filter(None,textwrap.dedent(EXPR).split("\n"))):
        print(expr)
        print(eval(expr))
    pass


    pl = pv.Plotter(window_size=SIZE*2)

    pl.add_points(pos)

    pl.show_grid()
    pl.show()


