#!/usr/bin/env python
"""
QBndTest.py
=============

::
 
   ipython -i tests/QBndTest.py
 

"""
import os 
import numpy as np

np.set_printoptions(suppress=True, precision=3, edgeitems=5 )

fold="$TMP/QBndTest"
load_ = lambda name:np.load(os.path.expandvars("%s/%s" % (fold,name)))

if __name__ == '__main__':
    src = load_("src.npy")
    dst = load_("dst.npy")

    s = src
    d = dst.reshape(src.shape)
    assert np.allclose( s, d )

    ss = src.reshape(dst.shape)
    dd = dst
    assert np.allclose( ss, dd )


