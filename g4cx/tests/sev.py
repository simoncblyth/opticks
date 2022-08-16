#!/usr/bin/env python
"""
sev.py
========

::

    In [11]: iid[hit_ii]
    Out[11]: 
    array([[29082,     2,  4938,  4938],
           [31842,     2,  8753,  8753],
           [32525,     2,  9656,  9656],
           [32355,     2,  9485,  9485],
           [35884,     2, 14395, 14395],
           ...,
           [40408,     3,  7566,  7566],
           [34641,     2, 12663, 12663],
           [29988,     2,  6174,  6174],
           [29182,     2,  5082,  5082],
           [37875,     2, 17165, 17165]], dtype=int32)

    In [18]: np.unique( iid[hit_ii,1], return_counts=True )
    Out[18]: (array([1, 2, 3], dtype=int32), array([ 138, 2587, 1068]))



"""

import numpy as np
from opticks.CSG.CSGFoundry import CSGFoundry
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree
from opticks.ana.eprint import eprint, epr

if __name__ == '__main__':


    cf = CSGFoundry.Load("$CFBASE", symbol="cf") 
    print(repr(cf))

    stf = Fold.Load("$STBASE/stree", symbol="stf" )
    st = stree(stf)
    print(repr(st))

    ev = Fold.Load("$FOLD", symbol="ev")
    print(repr(ev))

    hit_ii = ev.hit.view(np.int32)[:,1,3]  # sphoton.iindex
    print("hit_ii : %s " % str(hit_ii))

    iid = stf.inst_f4[:,:,3].view(np.int32)
    iid2 = cf.inst[:,:,3].view(np.int32)
    assert np.all( iid == iid2 ) 

    inst_diff = np.abs(stf.inst_f4[:,:,:3] - cf.inst[:,:,:3]).max() 
    print(" inst_diff : %s " % inst_diff )

     


