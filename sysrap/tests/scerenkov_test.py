#!/usr/bin/env python
"""
~/o/sysrap/tests/scerenkov_test.sh pdb 

"""

import os, numpy as np
MODE = int(os.environ.get("MODE",0))
from opticks.ana.fold import Fold

if MODE in [2,3]:
    from opticks.ana.pvplt import *
pass

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))



