#!/usr/bin/env python

import os, numpy as np
TEST = os.environ.get("TEST","") 
from opticks.ana.fold import Fold
from opticks.ana.p import *
PIDX = int(os.environ.get("PIDX","-1"))

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    if TEST == "makeGenstepArrayFromVector":
        assert np.all(t.igs == t.gs)
    pass 


    p = t.photon if hasattr(t, "photon") else None
    r = t.record if hasattr(t, "record") else None
    seq = t.seq if hasattr(t, "seq") else None
    nib = seqnib_(seq[:,0])  if not seq is None else None

    if p is None:
        print("p:None")
    else:
        for i in range(len(p)):
            if not (PIDX == -1 or PIDX == i): continue 
            if PIDX > -1: print("PIDX %d " % PIDX) 
            print("r[%d,:,:3]" % i)
            print(r[i,:nib[i],:3]) 
            print("\n\nbflagdesc_(r[%d,j])" % i)
            for j in range(nib[i]):
                print(bflagdesc_(r[i,j]))   
            pass

            print("\n") 
            print("p[%d]" % i)
            print(p[i])
            print("\n") 
            print("bflagdesc_(p[%d])" % i)
            print(bflagdesc_(p[i])) 
            print("\n")
            if not seq is None:
                print("seqhis_(seq[%d,0]) nib[%d]  " % (i,i) )
                print(" %s : %s "% ( seqhis_(seq[i,0]), nib[i] ))
                print("\n")
            pass
            print("\n\n")
        pass
    pass
pass


