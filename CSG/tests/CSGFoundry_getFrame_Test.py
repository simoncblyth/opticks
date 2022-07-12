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

    assert np.all( t.ipt1 == t.ipt2 )

    m2w = t.m2w[0] 

    #ipt3 = py_transform_1(t.ip, t.m2w[0] )

    ip = t.ip

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

    ipt3 = np.zeros( (len(ip),4,4) )
    ipt3[:,0] = tpos 
    ipt3[:,1] = tmom 
    ipt3[:,2] = tpol
 

    d_tpos = np.abs( ipt3[:,0,:3] - t.ipt1[:,0,:3] )
    d_tmom = np.abs( ipt3[:,1,:3] - t.ipt1[:,1,:3] )
    d_tpol = np.abs( ipt3[:,2,:3] - t.ipt1[:,2,:3] )

    print("d_tpos.max %s " % d_tpos.max() )
    print("d_tmom.max %s " % d_tmom.max() )
    print("d_tpol.max %s " % d_tpol.max() )



    normdiff_ = lambda v3:np.abs( 1. - np.sqrt(np.sum( np.power(v3, 2), axis=1)) ).max()   

    ipt0_mom_nd = normdiff_( t.ipt0[:,1,:3] )  
    ipt1_mom_nd = normdiff_( t.ipt1[:,1,:3] )  
    ipt2_mom_nd = normdiff_( t.ipt2[:,1,:3] )  
    ipt3_mom_nd = normdiff_( ipt3[:,1,:3] )  

    print("ipt0_mom_nd  %s" % ipt0_mom_nd  )
    print("ipt1_mom_nd  %s" % ipt1_mom_nd  )
    print("ipt2_mom_nd  %s" % ipt2_mom_nd  )
    print("ipt3_mom_nd  %s" % ipt3_mom_nd  )

    ipt0_pol_nd = normdiff_( t.ipt0[:,2,:3] )  
    ipt1_pol_nd = normdiff_( t.ipt1[:,2,:3] )  
    ipt2_pol_nd = normdiff_( t.ipt2[:,2,:3] )  
    ipt3_pol_nd = normdiff_( ipt3[:,2,:3] )  

    print("ipt0_pol_nd  %s" % ipt0_pol_nd  )
    print("ipt1_pol_nd  %s" % ipt1_pol_nd  )
    print("ipt2_pol_nd  %s" % ipt2_pol_nd  )
    print("ipt3_pol_nd  %s" % ipt3_pol_nd  )






