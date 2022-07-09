#!/usr/bin/env python 
"""
G4CXRenderTest.py
===================


  : t.isect                                            :      (1080, 1920, 4) : 0:00:34.591314 
  : t.photon                                           :   (1080, 1920, 4, 4) : 0:00:33.933167 


isect::

    In [9]: np.unique( t.isect[:,:,3].view(np.int32), return_counts=True )
    Out[9]: (array([ 65536, 131072], dtype=int32), array([1993260,   80340]))

    In [10]: 0xffff
    Out[10]: 65535

    In [11]: "%x" % 131072
    Out[11]: '20000'


The "frame photons" are unfilled::

    In [13]: t.photon[0,0]
    Out[13]: 
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)

    In [14]: np.all( t.photon == 0. )
    Out[14]: True

    In [15]: t.photon.shape
    Out[15]: (1080, 1920, 4, 4)


"""
import os, numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(t)


