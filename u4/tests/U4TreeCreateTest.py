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

    prim_nidx = f.prim_nidx[:,0] ## nidx values of every prim [globalPrimIdx(0-based, contiguous)]
    assert( np.all( prim_nidx > -1 ) )
    prim_nidx_tab = np.c_[np.unique(prim_nidx, return_counts=True)]
    assert( np.all( prim_nidx_tab[:,1] == 1 ) )

    nidx_prim = f.nidx_prim[:,0] ## prim values of every nidx [lots of repeats from instancing]
    assert( np.all( nidx_prim > -1 ) )
    nidx_prim_tab = np.c_[np.unique(nidx_prim, return_counts=True)]
    assert( len(nidx_prim_tab) == len(prim_nidx) )


    c = sn_check(f, symbol="c")
    print(repr(c))

    sn = f._csg.sn
    s_tv = f._csg.s_tv
    s_pa = f._csg.s_pa
    s_bb = f._csg.s_bb

    if hasattr(f, 'csg'):
        snd = f.csg.node
        xform = f.csg.xform
        param = f.csg.param
        aabb = f.csg.aabb

        EXPR = r"""
        np.all( snd == sn )
        np.all( xform == s_tv )
        np.all( param == s_pa )
        np.all( aabb == s_bb )
        """
    else:
        snd = None
        xform = None
        param = None
        aabb = None

        EXPR = r"""
        sn
        s_tv
        s_pa
        s_bb
        """
    pass

    for expr in list(filter(None,textwrap.dedent(EXPR).split("\n"))):
        print(expr)
        if expr[0] == "#": continue
        print(eval(expr))
    pass





