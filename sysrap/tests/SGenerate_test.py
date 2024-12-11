#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold 

MODE = int(os.environ.get("MODE",0))

if MODE in [2,3]:
    from opticks.ana.pvplt import *
pass

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    gs = f.gs 
    ph = f.ph 

    pos = ph[:,0,:3]
    mom = ph[:,1,:3]
    pol = ph[:,2,:3]

    if MODE == 3:
        pl = pvplt_plotter()
        #pl.add_points(pos, point_size=20)
        #pvplt_arrows(pl, pos, mom, factor=20 )

        pvplt_polarized(pl, pos, mom, pol, factor=20  )

        pl.show()
    pass
pass
