#!/usr/bin/env python

import numpy as np, textwrap
from opticks.ana.fold import Fold

if __name__ == '__main__':

    s = Fold.Load("$SFOLD", symbol="s")
    q = Fold.Load("$QFOLD", symbol="q")

    print(repr(s))
    print(repr(q))

    exprs = textwrap.dedent(r"""
    s.test.get_stackspec.shape
    q.stackspec_interp.shape 
    np.abs(s.test.get_stackspec-q.stackspec_interp).max()
    """).split("\n")

    for expr in exprs:
        print("[%s]"%expr)
        if len(expr) == 0: continue
        print(eval(expr))
    pass


  


