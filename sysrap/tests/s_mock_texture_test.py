#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    a = f.a 
    b = f.b 

    a[np.where(a==1e9)] = 1e6  # kludge for clearer repr 
    b[np.where(b==1e9)] = 1e6  

pass
