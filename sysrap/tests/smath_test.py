#!/usr/bin/env python

import os, textwrap, numpy as np
from opticks.ana.fold import Fold, EXPR_
from opticks.sysrap.smath import rotateUz

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    u = f.rotateUz[0,0]
    assert np.all( f.rotateUz[:,0] == u )

    d = f.rotateUz[:,1]
    d1 = f.rotateUz[:,2]
    d1p = rotateUz(d, u)

    for expr in EXPR_(r"""
u
d      # original direction from C++
d1     # C++ rotateUz
d1p    # py rotateUz
d1 - d1p
(d1 - d1p).min()
(d1 - d1p).max()
"""):
        print(expr)
        if expr == "" or expr[0] == "#": continue
        print(repr(eval(expr)))
    pass
pass


