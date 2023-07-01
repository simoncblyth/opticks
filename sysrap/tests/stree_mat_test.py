#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    if np.all( t.mat[16,0,:,1] == 1e9 ):
        print("Vacuum 1e9 kludge reduce to 1e6 : because it causes obnoxious presentation")
        t.mat[16,0,:,1] = 1e6 
    else:
        print("Not doing Vacuum kludge")
    pass

pass
 
