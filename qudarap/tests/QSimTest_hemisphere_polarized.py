#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.qudarap.tests.pvplt import pvplt_simple, pvplt_photon, pvplt_polarized
import pyvista as pv


FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")
TEST = os.environ["TEST"]





if __name__ == '__main__':
    t = Fold.Load(FOLD)

    lim = slice(0,1000)

    mom = t.p[lim,1,:3]
    pos = -mom     # hemisphere of photons all directed at origin 
    pol = t.p[lim,2,:3]

    print(mom) 
    print(pol) 

    pvplt_polarized( None, pos, mom, pol )

     


    

    






    
