#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
import pyvista as pv
COLORS = "cyan red green blue cyan magenta yellow pink orange purple lightgreen".split()


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    pp = t.pp

    os.environ["EYE"] = "-0.707,-100,0.707"
    os.environ["LOOK"] = "-0.707,0,0.707"

    label = "qsim_test.py "
    pl = pvplt_plotter(label=label)   

    lim = slice(None)

    mom0 = pp[:,0,1,:3]
    pol0 = pp[:,0,2,:3]

    mom1 = pp[:,1,1,:3]
    pol1 = pp[:,1,2,:3]

    pp[:,0,0,:3] = -mom0    # illustrative choice incident position on unit hemisphere
    pp[:,1,0,:3] = [0,0,0]  # illustrative choice transmitted position on unit hemisphere


    pvplt_viewpoint( pl ) 


    ii = [0,4,8,12]        # looks like the S-pol survives unscathed, but P-pol gets folded over

    ii = list(range(len(pp)))  
    

    for i in ii:
        polcol = COLORS[ i % len(COLORS)]
        pvplt_photon( pl, pp[i:i+1,0], polcol=polcol, polscale=0.5 )
        pvplt_photon( pl, pp[i:i+1,1], polcol=polcol, polscale=0.5 )
    pass

    cp = pl.show() 
    

    

