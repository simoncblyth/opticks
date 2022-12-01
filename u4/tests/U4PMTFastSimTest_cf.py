#!/usr/bin/env python
"""
U4PMTFastSimTest_cf.py
========================

::

    In [5]: seqhis_(a.seq[726,0])
    Out[5]: ['TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR', 'BT SR BT SA']

    In [6]: seqhis_(b.seq[726,0])
    Out[6]: ['TO BT BT SA', '?0?']


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
  


