#!/usr/bin/env python
"""

::

    In [5]: p.primBuf
    Out[5]: 
    array([[0, 1, 0, 3],
           [1, 7, 1, 0]], dtype=uint32)

    In [6]: p.partBuf.shape
    Out[6]: (8, 4, 4)

    In [7]: p.partBuf
    Out[7]: 
    array([[[    0.    ,     0.    ,     0.    ,  1000.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [-1000.01  , -1000.01  , -1000.01  ,     0.    ],
            [ 1000.01  ,  1000.01  ,  1000.01  ,     0.    ]],

           [[    0.    ,     0.    ,     0.    ,   500.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -500.01  ,  -500.01  ,  -500.01  ,     0.    ],
            [  500.01  ,   500.01  ,   500.01  ,     0.    ]],

"""
import os
import numpy as np

class GParts(object):
   def __init__(self, base):
       self.name = base
       self.partBuf = np.load(os.path.expandvars(os.path.join(base,"partBuffer.npy")))
       self.primBuf = np.load(os.path.expandvars(os.path.join(base,"primBuffer.npy")))
   pass
   def __getitem__(self, i):
       if i >= 0 and i < len(self.primBuf):
           prim = self.primBuf[i]
           partOffset, numPart, a, b = prim
           return self.partBuf[partOffset:partOffset+numPart]
       else:
           raise IndexError
       pass 


if __name__ == '__main__':
   p = GParts("/tmp/blyth/opticks")
   p0 = p[0]
   p1 = p[1]





