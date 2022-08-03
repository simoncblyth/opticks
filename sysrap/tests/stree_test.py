#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold
from opticks.CSG.CSGFoundry import CSGFoundry 


if __name__ == '__main__':
    t = Fold.Load()
    print(repr(t))

    cf = CSGFoundry.Load()
    print(cf)



