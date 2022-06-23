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
    a = Fold.Load("$A_FOLD", symbol="a")
    b = Fold.Load("$B_FOLD", symbol="b")
    assert (a.inphoton - b.inphoton).max() < 1e-10 

    # apply stag.Unpack to both as same stag.h bitpacking is used
    at = stag.Unpack(a.tag) if hasattr(a,"tag") else None
    bt = stag.Unpack(b.tag) if hasattr(b,"tag") else None


