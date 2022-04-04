#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    FOLD = os.environ["FOLD"]
    t = Fold.Load(FOLD)

    print(t.p[:,:3]) 
    print(t.p[:,3].view(np.uint32)) 


    
