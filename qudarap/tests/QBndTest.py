#!/usr/bin/env python
"""
QBndTest.py
=============

::
 
   ipython -i tests/QBndTest.py
 

"""
import os 
import numpy as np

from opticks.ana.fold import Fold

#np.set_printoptions(suppress=True, precision=3, edgeitems=5 )


if __name__ == '__main__':

    bt = Fold.Load("$TMP/QBndTest")
    src = bt.src    
    dst = bt.dst    


    s = src
    d = dst.reshape(src.shape)
    assert np.allclose( s, d )


    ss = src.reshape(dst.shape)
    dd = dst
    assert np.allclose( ss, dd )


