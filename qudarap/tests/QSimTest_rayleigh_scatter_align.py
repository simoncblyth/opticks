#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.qudarap.tests.pvplt import pvplt_simple, pvplt_photon
import pyvista as pv
import matplotlib.pyplot as plt 

GUI = not "NOGUI" in os.environ
SIZE = np.array([1280, 720])


FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")
TEST = os.environ["TEST"]

G4_FOLD = "/tmp/G4OpRayleighTest"



if __name__ == '__main__':

    #t = Fold.Load(FOLD)
    t = Fold.Load(G4_FOLD)

    print(t.p[:,:3]) 
    print(t.p[:,3].view(np.uint32)) 


    mom = t.p[:,1,:3]    
    pol = t.p[:,2,:3]    

    mom0 = t.p0[0,1,:3]
    pol0 = t.p0[0,2,:3]

    ct = np.sum( pol*pol0, axis=1 ) 


    transverse = np.sum( mom*pol, axis=1 )  
    tr_min, tr_max, tr_abs = transverse.min(), transverse.max(), np.abs(transverse).max() 
    print( " tr_min : %s  tr_max : %s tr_abs : %s " % ( tr_min, tr_max, tr_abs )) 
    assert tr_abs < 1e-5 


    mom_norm = np.abs(np.sum(mom*mom, axis=1 ) - 1.).max()
    pol_norm = np.abs(np.sum(pol*pol, axis=1 ) - 1.).max()

    print(" mom_norm : %s pol_norm %s " % (mom_norm, pol_norm))
    assert mom_norm < 1e-5
    assert pol_norm < 1e-5

    pl = pv.Plotter(window_size=SIZE*2 )  # retina 2x ?
    pl.show_grid()

    #pvplt_simple(pl, mom, "%s.mom" % TEST ) 

    pvplt_simple(pl, pol[:100000], "%s.pol" % TEST ) 
    pvplt_photon(pl,  t.p0 )

    cp = pl.show() if GUI else None 


    fig, ax = plt.subplots(1)   
    ax.hist( ct*ct, bins=100 )   
    fig.show()

 
