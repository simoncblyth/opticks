#!/usr/bin/env python 
"""
U4RecorderTest_ab.py
======================

Usage::

   cd ~/opticks/u4/tests   # u4t
   ./U4RecorderTest_ab.sh  

"""
import numpy as np

from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.sysrap.stag import stag  
from opticks.u4.U4Stack import U4Stack

np.set_printoptions(edgeitems=16) 



tag = stag()
stack = U4Stack()


if __name__ == '__main__':

    if "A_FOLD" in os.environ:
        a = Fold.Load("$A_FOLD", symbol="a")
        at = stag.Unpack(a.tag) if hasattr(a,"tag") else None
        ats = stag.StepSplit(at) if not at is None else None
    else:
        a = None 
    pass

    if "B_FOLD" in os.environ:
        b = Fold.Load("$B_FOLD", symbol="b")
        bt = stag.Unpack(b.tag) if hasattr(b,"tag") else None  # apply stag.Unpack to both as same stag.h bitpacking is used
        bts = stag.StepSplit(bt) if not bt is None else None
    else:
        b = None
    pass
    ab = (not a is None) and (not b is None)

    if ab: 
        assert (a.inphoton - b.inphoton).max() < 1e-10 
    pass



