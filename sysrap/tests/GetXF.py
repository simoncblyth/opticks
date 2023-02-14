#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

from opticks.sysrap.stree import snd

if __name__ == '__main__':
    t = Fold.Load("$FOLD/GetXF", symbol="t")
    print(repr(t))
pass


