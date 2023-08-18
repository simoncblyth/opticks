#!/usr/bin/env python
"""
U4TreeCreateSSimTest.py
=====================
"""

import numpy as np, textwrap
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree, snode

np.set_printoptions(edgeitems=16)

if __name__ == '__main__':
    f = Fold.Load("$FOLD/SSim/stree", symbol="f")
    print(repr(f))
    snode.Fields(bi=True)  # bi:True exports field indices into builtins scope

    EXPR = r"""
    f.nds.shape
    snode.Label(5,6)         # field label spacing and initial offset 
    ri,ro                    # (repeat_index, repeat_ordinal) 
    f.nds[f.nds[:,ri] == 1]  # structural nodes with repeat_index of 1 
    snode.Label(5,6)         # field label spacing and initial offset  
    snode.Doc()
    """

    for expr in list(filter(None,textwrap.dedent(EXPR).split("\n"))):
        print(expr)
        if len(expr) == 0 or expr[0] == "#": continue
        print(eval(expr))
    pass





