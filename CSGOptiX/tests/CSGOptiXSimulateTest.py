#!/usr/bin/env python
import sys, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 
PIDX = int(os.environ.get("PIDX","-1"))

if __name__ == '__main__':
    t = Fold.Load()
    r = t.record if hasattr(t,'record') else None
    p = t.photon if hasattr(t,'photon') else None

    if p is None:
       print("ERROR p is None")
       sys.exit(0)
    pass 

    s = str(p[:,:3])
    a = np.array( s.split("\n") + [""] ).reshape(-1,4)


    for i in range(len(a)):
        if not (PIDX == -1 or PIDX == i): continue
        if PIDX > -1: print("PIDX %d " % PIDX)

        if not r is None:
            print("r[i,:,:3]")
            print(r[i,:,:3])
            print("\n\nbflagdesc_(r[i,j])")
            for j in range(len(r[i])):
                print(bflagdesc_(r[i,j])  )
            pass
        pass

        print("\n")
        print("p")
        print("\n".join(a[i]))
        print(bflagdesc_(p[i]))
        print("\n")




