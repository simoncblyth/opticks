#!/usr/bin/env python 
"""
stree_sensor_test.py
======================

Count the repeat_index nodes, checking consistent with product of instance count 
and subtree nodes within each instance::

    In [4]: np.unique( st.nds.repeat_index, return_counts=True )
    Out[4]: 
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32),
     array([  3089, 128000,  88305,  34979,  14400,    590,    590,    590,    590,  65520]))

    In [7]: st.f.factor[:,:4]
    Out[7]: 
    array([[    0, 25600, 25600,     5],
           [    1, 12615, 12615,     7],
           [    2,  4997,  4997,     7],
           [    3,  2400,  2400,     6],
           [    4,   590,     0,     1],
           [    5,   590,     0,     1],
           [    6,   590,     0,     1],
           [    7,   590,     0,     1],
           [    8,   504,     0,   130]], dtype=int32)

    In [10]: np.c_[st.f.factor[:,:4], st.f.factor[:,1]*st.f.factor[:,3]]
    Out[10]: 
    array([[     0,  25600,  25600,      5, 128000],
           [     1,  12615,  12615,      7,  88305],
           [     2,   4997,   4997,      7,  34979],
           [     3,   2400,   2400,      6,  14400],
           [     4,    590,      0,      1,    590],
           [     5,    590,      0,      1,    590],
           [     6,    590,      0,      1,    590],
           [     7,    590,      0,      1,    590],
           [     8,    504,      0,    130,  65520]], dtype=int32)

"""
import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree 

if __name__ == '__main__':
    f = Fold.Load("$BASE/stree", symbol="f")
    st = stree(f)
    print(repr(st))
pass
