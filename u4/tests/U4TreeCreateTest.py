#!/usr/bin/env python 
"""
U4TreeCreateTest.py
=====================
"""

import numpy as np, textwrap
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

    s_tv = f._csg.s_tv
    xform = f.csg.xform

    s_pa = f._csg.s_pa 
    param = f.csg.param

    s_bb = f._csg.s_bb
    aabb = f.csg.aabb 
 

    EXPR = r"""
    np.all( snd == sn )
    np.all( xform == s_tv ) 
    np.all( param == s_pa )
    np.all( aabb == s_bb )
    """

    for expr in list(filter(None,textwrap.dedent(EXPR).split("\n"))):
        print(expr)
        if expr[0] == "#": continue
        print(eval(expr))
    pass   

      



