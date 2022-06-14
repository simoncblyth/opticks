#!/usr/bin/env python 
"""
U4RecorderTest_ab.py
======================

Usage::

   cd ~/opticks/u4/tests
   ./U4RecorderTest_ab.sh  

"""
import numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 

if __name__ == '__main__':
    a = Fold.Load("$A_FOLD", symbol="a")
    b = Fold.Load("$B_FOLD", symbol="b")
    assert (a.inphoton - b.inphoton).max() < 1e-10 






