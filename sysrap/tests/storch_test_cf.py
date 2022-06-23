#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    a = Fold.Load("$A_FOLD", symbol="a")
    b = Fold.Load("$B_FOLD", symbol="b")

    print(a)
    print(b)

    assert np.all( a.gs == b.gs )
    assert np.all( a.se == b.se )
    assert np.all( a.ph == b.ph )

