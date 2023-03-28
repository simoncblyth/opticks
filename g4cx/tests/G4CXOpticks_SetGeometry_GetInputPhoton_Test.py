#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
pass

