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
flagdesc_ = lambda p:" %4d %10s id:%6d ori:%d idx:%6d %10s " % ( boundary_(p), hm.label(flag_(p)), identity_(p), orient_(p), idx_(p), hm.label( flagmask_(p) ))


if __name__ == '__main__':
    t = Fold.Load()

    p = t.p
    r = t.r
    prd = t.prd

    s = str(p[:,:3]) 
    a = np.array( s.split("\n") + [""] ).reshape(-1,4)

    for i in range(len(a)):

        print(r[i,:,:3]) 
        for j in range(len(r[i])):
            print(flagdesc_(r[i,j])) 
        pass

        print("\n") 
        print("\n".join(a[i]))
        print(flagdesc_(p[i]))
        print("\n") 

        print("\n\n") 
    pass



    
