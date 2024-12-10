#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold 
from opticks.ana.pvplt import *
MODE = int(os.environ.get("MODE",0))

if __name__ == '__main__':
    print("MODE:%d" % MODE)
    f = Fold.Load(symbol="f")
    print(repr(f))

    gs = f.gs 
    se = f.se 
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
    else:
        pass
    pass
