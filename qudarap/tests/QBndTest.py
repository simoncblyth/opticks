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


def test_cf(src, dst):
    s = src
    d = dst.reshape(src.shape)
    assert np.allclose( s, d )

    ss = src.reshape(dst.shape)
    dd = dst
    assert np.allclose( ss, dd )



if __name__ == '__main__':

    t = Fold.Load("$TMP/QBndTest/Add")
    src = t.src    
    dst = t.dst    



