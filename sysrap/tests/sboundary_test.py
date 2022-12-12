#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    a = Fold.Load("$AFOLD", symbol="a")
    b = Fold.Load("$BFOLD", symbol="b")

    print(repr(a))
    print(repr(b))
pass

