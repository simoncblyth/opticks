#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 

if __name__ == '__main__':
    t = Fold.Load(); 
    PIDX = int(os.environ.get("PIDX","-1"))
    
    p = t.photon if hasattr(t, "photon") else None
    r = t.record if hasattr(t, "record") else None
    seq = t.seq if hasattr(t, "seq") else None


    for i in range(len(p)):
        if not (PIDX == -1 or PIDX == i): continue 
        if PIDX > -1: print("PIDX %d " % PIDX) 
        print("r[%d,:,:3]" % i)
        print(r[i,:,:3]) 
        print("\n\nbflagdesc_(r[%d,j])" % i)
        for j in range(len(r[i])):
            print(bflagdesc_(r[i,j]))   
        pass

        print("ridiff_(r[%d])*1000." % i)
        print(ridiff_(r[i])*1000.)   

        print("\n") 
        print("p[%d]" % i)
        print(p[i])
        print("\n") 
        print("bflagdesc_(p[%d])" % i)
        print(bflagdesc_(p[i])) 
        print("\n") 
        if not seq is None:
            print("seqhis_(seq[%d,0])" % i) 
            print(seqhis_(seq[i,0])) 
            print("\n") 
        pass
        print("\n\n") 
    pass





