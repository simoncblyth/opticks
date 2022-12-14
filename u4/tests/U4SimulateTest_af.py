#!/usr/bin/env python
"""
U4SimulateTest_allcf.py
========================

"""


import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 

MODE = int(os.environ.get("MODE", 0))
if MODE > 0:
    from opticks.ana.pvplt import * 
pass

PID = int(os.environ.get("PID", -1))
if PID == -1: PID = int(os.environ.get("OPTICKS_G4STATE_RERUN", -1))


if __name__ == '__main__':

    print("PID : %d " % (PID))
    a = Fold.Load("$BASE/ALL0", symbol="a")
    print(repr(a))

    b = Fold.Load("$BASE/ALL1", symbol="b")
    print(repr(b))

    print("seqhis_(a.seq[PID,0] : %s " % seqhis_(a.seq[PID,0] ))
    print("seqhis_(b.seq[PID,0] : %s " % seqhis_(b.seq[PID,0] ))
  
    ar = a.record[PID]
    br = b.record[PID]

    print("\nar = a.record[PID] ; ar[:4] \n", ar[:4])
    print("\nbr = b.record[PID] ; br[:4] \n", br[:4] )


