#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")

if __name__ == '__main__':
    t = Fold.Load(FOLD)

    print(t.p) 

    flag = t.p[:,3,3].view(np.uint32)
    print(np.unique(flag, return_counts=True))



    
