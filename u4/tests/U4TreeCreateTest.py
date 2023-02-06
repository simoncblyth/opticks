#!/usr/bin/env python 
"""
U4TreeCreateTest.py
=====================

In [5]: np.unique( t.nds.reshape(-1,4,4)[:,3,0], return_counts=True )                                                                                     
Out[5]: 
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], dtype=int32),
 array([    1,     3,     1,     2,     1,     2,   191,     1,     1,   504,   504, 32256, 32256,     1,     1,  2120,     1,     1,  3048,     1,    46,     8,   370,   220, 27370,    56, 43213,
        17612,  4997, 45612, 20012,  4997,  4997, 12615, 12615, 12615, 25600, 25600,     1,     1,  2400,  2400,  2400]))




"""

import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree, snode

np.set_printoptions(edgeitems=16)

if __name__ == '__main__':
    f = Fold.Load("$FOLD/stree", symbol="f")
    print(repr(f))

    snode.Fields(bi=True)  # bi:True exports field indices into builtins scope

    print(snode.Label(6,11),"\n", f.nds[f.nds[:,ri] == 1 ])




