#!/usr/bin/env python
"""
stree_load_test_tmpfold.py
===========================


"""

import numpy as np
from opticks.ana.fold import Fold

np.set_printoptions(edgeitems=16)


if __name__ == '__main__':
    f = Fold.Load("$TMPFOLD/$TEST", symbol="f")
    print(repr(f))
pass

