#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
from opticks.ana.p import * 

if __name__ == '__main__':
    t = Fold.Load(globals=True)
    PIDX = int(os.environ.get("PIDX","-1"))

    lim = slice(0, 1000)

    pos = t.p[lim,0,:3]
    mom = t.p[lim,1,:3]
    pol = t.p[lim,2,:3]

    pl = pvplt_plotter()
    pvplt_polarized(pl, pos, mom, pol, factor=20  )
    pl.show()
  


   
