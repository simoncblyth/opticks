#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")

if __name__ == '__main__':
    t = Fold.Load(FOLD)

    p = t.p 

    print(p) 

    flag = p[:,3,3].view(np.uint32)
    print(np.unique(flag, return_counts=True))

    TransCoeff = p[:,1,3]
    print( "TransCoeff %s " %  TransCoeff  )

    flat = p[:,0,3]
    print(" flat %s " % flat)





    
