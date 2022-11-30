#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold


if __name__ == '__main__':
    t = Fold.Load("$FOLD/test_U4UniformRand", symbol="t")
    print(repr(t))
    assert np.all( t.u0 == t.u1 )
    assert np.all( t.u2 == t.u1 )


