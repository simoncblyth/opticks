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

    ddist_ = lambda a,i:np.sqrt(np.sum( (a.record[:,i+1,0,:3]-a.record[:,i,0,:3])*(a.record[:,i+1,0,:3]-a.record[:,i,0,:3]) , axis=1 ))
    dtime_ = lambda a,i:a.record[:,i+1,0,3] - a.record[:,i,0,3]  
    dspeed_ = lambda a,i:ddist_(a,i)/dtime_(a,i)








