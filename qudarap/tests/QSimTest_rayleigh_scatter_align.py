#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")

if __name__ == '__main__':
    t = Fold.Load(FOLD)

    print(t.p[:,:3]) 
    print(t.p[:,3].view(np.uint32)) 


    
