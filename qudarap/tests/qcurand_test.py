#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    print("f.f\n",f.f)
    print("f.d\n",f.d)

pass

