#!/usr/bin/env python
"""
U4SimulateTest_cf.py
========================

::

    PID = 726    
    seqhis_(a.seq[PID,0] : ['TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR', 'BT SR BT SA'] 
    seqhis_(b.seq[PID,0] : ['TO BT BT SR SR BR BR SR SR SR BR SR BR SR SA', '?0?'] 



"""


import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 

NOGUI = "NOGUI" in os.environ
MODE = int(os.environ.get("MODE", 0))
if not NOGUI:
    from opticks.ana.pvplt import * 
pass

PID = int(os.environ.get("PID", -1))
if PID == -1: PID = int(os.environ.get("OPTICKS_G4STATE_RERUN", -1))

if __name__ == '__main__':

    print("PID : %d " % (PID))
    a = Fold.Load("$BASE/SEL0", symbol="a")
    print(repr(a))

    b = Fold.Load("$BASE/SEL1", symbol="b")
    print(repr(b))

    print("seqhis_(a.seq[PID,0] : %s " % seqhis_(a.seq[PID,0] ))
    print("seqhis_(b.seq[PID,0] : %s " % seqhis_(b.seq[PID,0] ))
  
    ar = a.record[PID]
    br = b.record[PID]

    # mapping from new to old point index for PID 726 big bouncer
    b2a = np.array([ 0,1,3,5,6,8,9,11,12,13,15,17,19 ])
    abr = np.zeros( (len(b2a), 2, 4), dtype=np.float32 )
    for bidx,aidx in enumerate(b2a):
        abr[bidx,0] = ar[aidx,0]
        abr[bidx,1] = br[aidx,0]
    pass    
    print(repr(abr))

