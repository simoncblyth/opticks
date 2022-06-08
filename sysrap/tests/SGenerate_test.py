#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold 
from opticks.ana.pvplt import *

if __name__ == '__main__':
    t = Fold.Load()
    gs = t.gs 
    se = t.se 
    ph = t.ph 

    pos = ph[:,0,:3]
    mom = ph[:,1,:3]
    pol = ph[:,2,:3]

    pl = pvplt_plotter()
    #pl.add_points(pos, point_size=20)
    #pvplt_arrows(pl, pos, mom, factor=20 )

    pvplt_polarized(pl, pos, mom, pol, factor=20  )

    pl.show()
