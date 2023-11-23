#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold 
from opticks.ana.pvplt import *

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    gs = f.gs 
    se = f.se 
    ph = f.ph 


    lim = slice(0,1000)

    print(" ph %s lim %s " % ( str(ph.shape), str(lim)) )
    pos = ph[:,0,:3]
    mom = ph[:,1,:3]
    pol = ph[:,2,:3]

    expr = "np.sum(pol*mom,axis=1).max() # check transverse "
    print(expr)
    print(eval(expr)) 


    pl = pvplt_plotter()
    #pl.add_points(pos, point_size=20)
    #pvplt_arrows(pl, pos, mom, factor=20 )

    pvplt_polarized(pl, pos[lim], mom[lim], pol[lim], factor=20  )

    


    pl.show()



