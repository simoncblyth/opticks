#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold


from opticks.ana.hismask import HisMask   
hm = HisMask()

boundary_ = lambda p:p.view(np.uint32)[3,0] >> 16
flag_     = lambda p:p.view(np.uint32)[3,0] & 0xffff
identity_ = lambda p:p.view(np.uint32)[3,1]   
idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff
orient_   = lambda p:p.view(np.uint32)[3,2] >> 31
flagmask_ = lambda p:p.view(np.uint32)[3,3]
flagdesc_ = lambda p:" %6d prd(%3d %3d %1d)  %3s  %10s " % ( idx_(p),  boundary_(p),identity_(p),orient_(p),  hm.label(flag_(p)),hm.label( flagmask_(p) ))


if __name__ == '__main__':
    t = Fold.Load()

    p = t.p
    r = t.r
    prd = t.prd

    s = str(p[:,:3]) 
    a = np.array( s.split("\n") + [""] ).reshape(-1,4)

    for i in range(len(a)):

        print("r")
        print(r[i,:,:3]) 
        print("\n\nflagdesc_")
        for j in range(len(r[i])):
            print(flagdesc_(r[i,j])) 
        pass

        print("\n") 
        print("p")
        print("\n".join(a[i]))
        print(flagdesc_(p[i]))
        print("\n") 

        print("\n\n") 
    pass



    
