#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import *
from opticks.ana.histype import HisType

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    p = f.photon
    r = f.record
    s = f.seq
    h = f.hit 


 



