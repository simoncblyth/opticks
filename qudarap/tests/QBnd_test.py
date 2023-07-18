#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold 

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    f.src[np.where(f.src == 1e9)] = 1e6
    f.dst[np.where(f.dst == 1e9)] = 1e6

    a = f.src
    b = f.dst

pass
