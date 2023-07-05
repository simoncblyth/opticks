#!/usr/bin/env python

import numpy as np, textwrap
from opticks.ana.fold import Fold
import matplotlib.pyplot as plt
SIZE = np.array([1280,720])

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    s = Fold.Load("$FOLD/stree", symbol="s")
    print(repr(s))





