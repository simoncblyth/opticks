#!/usr/bin/env python 
"""
U4TreeCreateTest.py
=====================
"""

import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree, snode
from opticks.sysrap.sn_check import sn_check

np.set_printoptions(edgeitems=16)

if __name__ == '__main__':
    f = Fold.Load("$FOLD/stree", symbol="f")
    print(repr(f))
    snode.Fields(bi=True)  # bi:True exports field indices into builtins scope
    print(snode.Label(6,11),"\n", f.nds[f.nds[:,ri] == 1 ])

    c = sn_check(f, symbol="c") 
    print(repr(c))

    sn = f._csg.sn   
    snd = f.csg.node   




