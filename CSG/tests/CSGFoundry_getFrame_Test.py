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


def py_transform_0(ip, m2w):

    ipt = np.zeros( (len(ip),4,4) )
    ipt[:,0,:3] = ip[:,0,:3]
    ipt[:,0,3]  = 1.      # pos transformed as position 

    ipt[:,1,:3] = ip[:,1,:3]
    ipt[:,1,3]  = 0.      # mom tranformed as direction

    ipt[:,2,:3] = ip[:,2,:3]
    ipt[:,2,3]  = 0.      # pol transformed as direction 


    ipt[:,0] = np.dot( ipt[:,0], m2w )
    ipt[:,1] = np.dot( ipt[:,1], m2w )
    ipt[:,2] = np.dot( ipt[:,2], m2w )

    return ipt 

def py_transform_1(ip, m2w):

    pos = np.zeros( (len(ip), 4) ) 
    mom = np.zeros( (len(ip), 4) ) 
    pol = np.zeros( (len(ip), 4) ) 

    pos[:,:3] = ip[:,0,:3] 
    pos[:,3] = 1.

    mom[:,:3] = ip[:,1,:3] 
    mom[:,3] = 0.

    pol[:,:3] = ip[:,2,:3] 
    pol[:,3] = 0.

    tpos = np.dot( pos, m2w )
    tmom = np.dot( mom, m2w )
    tpol = np.dot( pol, m2w )

    ipt = np.zeros( (len(ip),4,4) )
    ipt[:,0] = tpos 
    ipt[:,1] = tmom 
    ipt[:,2] = tpol
    return ipt 



if __name__ == '__main__':
    t = Fold.Load()
    print(t)
    print(repr(t))

    m2w = t.m2w[0] 
    ip = t.ip          # local frame input photons 

    pos = np.zeros( (len(ip), 4) ) 
    mom = np.zeros( (len(ip), 4) ) 
    pol = np.zeros( (len(ip), 4) ) 

    pos[:,:3] = ip[:,0,:3] ; pos[:,3] = 1.    # position 
    mom[:,:3] = ip[:,1,:3] ; mom[:,3] = 0.    # direction
    pol[:,:3] = ip[:,2,:3] ; pol[:,3] = 0.    # direction

    tpos = np.dot( pos, m2w )
    tmom = np.dot( mom, m2w )
    tpol = np.dot( pol, m2w )

    tpho = np.zeros( (len(ip),4,4) )
    tpho[:,0] = tpos
    tpho[:,1] = tmom
    tpho[:,2] = tpol


   
 
    ipt = {} 
    ipt[0] = t.ipt0 
    ipt[1] = t.ipt1 
    ipt[2] = t.ipt2 
    ipt[3] = tpho

    lpt = {}
    lpt[0] = "ipt0 : C++ sframe transformed input photons with normalize:false "
    lpt[1] = "ipt1 : C++ sframe transformed input photons with normalize:true "
    lpt[2] = "ipt2 : C++ SEvt::getInputPhotons transformed input photons with normalize:true "
    lpt[3] = "ipt3 : python transformed input photons "

  
    for i in range(4):
        print(lpt[i])
    pass

    print("cross comparison (q:3,i:4,j:4) cross comparison  ")
    print("HUH: ipt2 is standing out as worst position match " )
    print("EXPECTED ipt2 and ipt1 to be perfect match : but they are not ??? ")

    qij = np.zeros( (3,4,4) )
    for q in range(3):
        for i in range(4):
            for j in range(4):
                _qij = np.abs( ipt[i][:,q,:3] - ipt[j][:,q,:3] )
                qij[q,i,j] = _qij.max() 
            pass
        pass
    pass
    print("q:pos,mom,pol ")
    print("qij*1e6:\n",qij*1e6)


    normdiff_ = lambda v3:np.abs( 1. - np.sqrt(np.sum( np.power(v3, 2), axis=1)) ).max()   

    nd = np.zeros( (3,4) )
    for q in np.arange(1,3):
        for i in range(4):
            nd[q,i] = normdiff_( ipt[i][:,q,:3] ) 
        pass
    pass
    print("\nnormdiff_ difference from 1. (pos dummy zero)  ")
    print("nd*1e6:\n", nd*1e6)


