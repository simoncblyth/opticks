#!/usr/bin/env python

import os, numpy as np
from opticks.sysrap.sevt import SEvt 
from opticks.ana.fold import Fold

if __name__ == '__main__':
    if "OLD" in os.environ:
        f = Fold.Load(symbol="f")
        print(repr(f))
    else:
        a = SEvt.Load("$FOLD",symbol="a")
        print(repr(a))
    pass

