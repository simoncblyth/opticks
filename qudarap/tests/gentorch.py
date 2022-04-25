#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
from opticks.ana.p import * 

if __name__ == '__main__':

    t = Fold.Load()
    PIDX = int(os.environ.get("PIDX","-1"))

    p= t.p 

    pos = p[:,0,:3]
    mom = p[:,1,:3]
    pol = p[:,2,:3]

    pl = pvplt_plotter()
    pvplt_polarized(pl, pos, mom, pol, factor=20  )
    pl.show()
    



