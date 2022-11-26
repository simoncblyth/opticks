#!/usr/bin/env python
"""
morton2d.py
=============

Morton coarsening expts 

"""

import numpy as np
from opticks.npy.mortonlib.morton2d import morton2d

sbitmask_ = lambda i:np.uint64(-1) >> np.uint64(64-i)


class Domain2D(object):
    def __init__(self, xyz, axes=(0,2)):
        pass


if __name__ == '__main__':

    phi = np.arange( 0, 2*np.pi, 2*np.pi/100 )
    xyz = np.zeros( (len(phi),3), dtype=np.float32 )

    xyz[:,0] = np.cos(phi)
    xyz[:,1] = 0.
    xyz[:,2] = np.sin(phi)

    axes = (0,2)

    H, V = axes 

    h = xyz[:,H]
    v = xyz[:,V]

    hdom = np.array( [ h.min(), h.max() ] )
    vdom = np.array( [ v.min(), v.max() ] )
    
    # scale coordinates into 0->1  
    sh = (h - hdom[0])/(hdom[1]-hdom[0])  
    sv = (v - vdom[0])/(vdom[1]-vdom[0])  

    scale = 0xffffffff
              
    ## 32bit range integer coordinates stored in 64 bit  
    ih = np.array( sh*scale, dtype=np.uint64 ) 
    iv = np.array( sv*scale, dtype=np.uint64 ) 

    khv = morton2d.Key(ih, iv)  ## morton interleave the two coordinates into one 64 bit code

    nbit = 8 
    mask = ~sbitmask_(64-nbit)
    print( "nbit {0:2d} mask {1:064b} {2:016x}  ".format(nbit, mask, mask) )    

    r_khv = khv & np.uint64(mask) 
    u_khv, i_khv, c_khv = np.unique( r_khv , return_index=True, return_counts=True)
    u_h, u_v = morton2d.Decode(u_khv) 


    #.astype(np.float32)/np.float32(0xffffffff)




 
