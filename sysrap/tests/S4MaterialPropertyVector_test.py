#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load("$FOLD/$TEST", symbol="f")
    print(repr(f))
pass

