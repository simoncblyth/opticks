#!/usr/bin/env python

import os, numpy as np

from opticks.ana.fold import Fold
from opticks.ana.pvplt import *

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    a = f.SmearNormal 
    n = f.SmearNormal_names
    sigma_alpha = f.SmearNormal_meta.sigma_alpha[0] 
    polish = f.SmearNormal_meta.polish[0] 

    label = "white:%s sigma_alpha:%s       red:%s polish:%s " % ( n[0],sigma_alpha, n[1],polish ) 
    pl = pvplt_plotter(label=label)
    pvplt_viewpoint( pl )


    pos = np.array( [[0,0,0]] )
    vec = np.array( [[0,0,1]] ) 
    pvplt_lines( pl, pos, vec )

    pl.add_points( a[:,0] , color="white" )
    pl.add_points( a[:,1] , color="red" )

    cpos = pl.show()






