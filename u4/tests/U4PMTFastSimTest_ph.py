#!/usr/bin/env python
"""
U4PMTFastSimTest_ph.py
========================

::

    u4t
    ./U4PMTFastSimTest.sh ph
    ./U4PMTFastSimTest.sh nph



"""
import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 

LABEL = os.environ.get("LABEL", "U4PMTFastSimTest_ph.py" )
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,3]
PID = int(os.environ.get("PID", -1))
if PID == -1: PID = int(os.environ.get("OPTICKS_G4STATE_RERUN", -1))

if MODE > 0:
    from opticks.ana.pvplt import * 
pass

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    print( " MODE : %d " % (MODE) )

    axes = X, Z 
    H,V = axes 
    label = LABEL 

    #pos = t.photon[:,0,:3]
    pos = t.record[:,:,0,:3].reshape(-1,3) 

    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        fig, ax = mpplt_plotter(label)
        ax.scatter( pos[:,H], pos[:,V], s=1 )  
        fig.show()
    elif MODE == 3:
        pl = pvplt_plotter(label)
        os.environ["EYE"] = "0,100,165"
        os.environ["LOOK"] = "0,0,165"
        pvplt_viewpoint(pl)
        pl.add_points( pos )
        pl.show()
    pass
pass
