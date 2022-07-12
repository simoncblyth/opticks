#!/usr/bin/env python
"""

Majority have position differences at 4e-4 level::

    In [23]: np.unique(np.where( d_tpos > 4e-4 )[0]).shape
    Out[23]: (4447,)

    In [24]: np.unique(np.where( d_tpos > 5e-4 )[0]).shape
    Out[24]: (0,)

"""

import os, numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load()
    print(t)

    assert np.all( t.ipt == t.ipt2 )

    pos = np.zeros( (len(t.ip), 4) ) 
    mom = np.zeros( (len(t.ip), 4) ) 
    pol = np.zeros( (len(t.ip), 4) ) 

    pos[:,:3] = t.ip[:,0,:3] 
    pos[:,3] = 1.

    mom[:,:3] = t.ip[:,1,:3] 
    mom[:,3] = 0.

    pol[:,:3] = t.ip[:,2,:3] 
    pol[:,3] = 0.


    m2w = t.m2w[0] 

    tpos = np.dot( pos, m2w )
    tmom = np.dot( mom, m2w )
    tpol = np.dot( pol, m2w )


    d_tpos = np.abs( tpos[:,:3] - t.ipt[:,0,:3] )
    d_tmom = np.abs( tmom[:,:3] - t.ipt[:,1,:3] )
    d_tpol = np.abs( tpol[:,:3] - t.ipt[:,2,:3] )

    print("d_tpos.max %s " % d_tpos.max() )
    print("d_tmom.max %s " % d_tmom.max() )
    print("d_tpol.max %s " % d_tpol.max() )





