#!/usr/bin/env python

import numpy as np, textwrap
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    snd = f.csg.node    
    sn = f._csg.sn

    EXPR = list(filter(None, textwrap.dedent(r"""
    snd[:,:11]
    sn 
    np.all( snd[:,:11] == sn )
    """).split("\n")))

    for expr in EXPR:
        print(expr)
        if expr[0] == "#": continue
        print(eval(expr))
    pass

pass
