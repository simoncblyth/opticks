#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f0 = Fold.Load("$FOLD0",symbol="f0")
    print(repr(f0))

    f1 = Fold.Load("$FOLD1",symbol="f1")
    print(repr(f1))
  
