#!/usr/bin/env python

import os, numpy as np
MODE = int(os.environ.get("MODE","0"))

PIDX = int(os.environ.get("PIDX","-1"))

from opticks.ana.fold import Fold
from opticks.ana.p import * 

if MODE in [2,3]:
    from opticks.ana.pvplt import *
pass

if __name__ == '__main__':
    t = Fold.Load(symbol="t",globals=True)
    print(repr(t))

    lim = slice(0, 1000)

    pos = t.p[lim,0,:3]
    mom = t.p[lim,1,:3]
    pol = t.p[lim,2,:3]

    print("t.p.shape\n", t.p.shape)
    print("pos.shape\n", pos.shape)
    print("mom.shape\n", mom.shape)
    print("pol.shape\n", pol.shape)

    if MODE == 3:
        pl = pvplt_plotter()
        pvplt_polarized(pl, pos, mom, pol, factor=20  )
        pl.show()
    pass
      


   
