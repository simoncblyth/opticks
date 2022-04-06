#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load()

    p = t.p
    r = t.r
    prd = t.prd


    print(p[:,:3]) 
    print(p[:,3].view(np.uint32)) 


    
