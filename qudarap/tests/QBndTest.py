#!/usr/bin/env python
"""
QBndTest.py
=============

::
 
   ./QBndTest.sh 
 

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
    f = Fold.Load(symbol="f")
    print(repr(f))

    f.src[np.where(f.src == 1e9)] = 1e6 
    f.dst[np.where(f.dst == 1e9)] = 1e6 

    a = f.src    
    b = f.dst    

    assert( np.all( a == b ))



