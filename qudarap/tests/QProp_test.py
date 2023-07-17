#!/usr/bin/env python 

import numpy as np
from opticks.ana.fold import Fold
from opticks.qudarap.tests.QPropTest import QPropTest

if __name__ == '__main__':
    f = Fold.Load("$FOLD/float", symbol="f")
    print(repr(f))

    t = QPropTest(f)
    t.plot()

pass

