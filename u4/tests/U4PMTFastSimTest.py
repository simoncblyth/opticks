#!/usr/bin/env python
"""
U4PMTFastSimTest.py
=====================

TODO: restrict to 2D so can create some more informative plots 



::

    In [10]: np.unique(wai, return_counts=True)
    Out[10]: (array([1, 2]), array([32628, 67372]))

    In [2]: np.unique( np.logical_and( wai == 1, trg == 0 ), return_counts=True )
    Out[2]: (array([False,  True]), array([67436, 32564]))

    In [3]: np.unique( np.logical_and( wai == 1, trg == 1 ), return_counts=True )
    Out[3]: (array([False,  True]), array([99936,    64]))

    In [4]: np.unique( np.logical_and( wai == 2, trg == 0 ), return_counts=True )
    Out[4]: (array([False,  True]), array([83308, 16692]))

    In [5]: np.unique( np.logical_and( wai == 2, trg == 1 ), return_counts=True )
    Out[5]: (array([False,  True]), array([49320, 50680]))




"""
import os, numpy as np
from opticks.ana.fold import Fold
NOGUI = "NOGUI" in os.environ
MODE = int(os.environ.get("MODE", 0))
if not NOGUI:
    from opticks.ana.pvplt import * 
pass

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    fs = t.SFastSim_Debug   
    kInfinity = 9.000e+99    

    pos = fs[:,0,:3]
    tim = fs[:,0,3]

    mom = fs[:,1,:3]
    ds1 = fs[:,1,3]

    pol = fs[:,2,:3]
    ds2 = fs[:,2,3]

    trg = fs[:,3,0].astype(np.int64)
    wai = fs[:,3,1].astype(np.int64)
    c   = fs[:,3,2]
    d   = fs[:,3,3]

    ms1 = np.where(ds1 == kInfinity )[0]
    ht1 = np.where(ds1 != kInfinity )[0]
    ms2 = np.where(ds2 == kInfinity )[0]
    ht2 = np.where(ds2 != kInfinity )[0]
    
    trg_no  = np.where(trg == 0)[0]
    trg_yes = np.where(trg == 1)[0]

    wai_1   = np.where( wai == 1)[0]
    wai_2   = np.where( wai == 2)[0]

    wai_1_trg_1 = np.logical_and( wai == 1, trg == 1 )   # only small number of these, all at top
    yellow = wai_1_trg_1 

    label = "MODE:%d " % MODE 
    if MODE == 0:
        blue = trg_no
        cyan = trg_yes
        label += "blue:trg_no "
        label += "cyan:trg_yes "
    elif MODE == 1:
        blue = wai_1
        cyan = wai_2
        label += "blue:wai_1 "
        label += "cyan:wai_2 "
    pass

    if NOGUI:
        print("not plotting as NOGUI in environ")
    else:
        pl = pvplt_plotter("U4PMTFastSimTest.py:SFastSim_Debug " + label  )
        #pl.add_points( pos[blue], color="blue" )
        #pl.add_points( pos[cyan], color="cyan" )
        pl.add_points( pos[yellow], color="yellow" )
        pl.show()
    pass


pass
