#!/usr/bin/env python
"""
"""
import os, textwrap, numpy as np
from opticks.ana.fold import Fold

expr = list(filter(None,textwrap.dedent(r"""
    np.all( a.prd == b.prd )
    np.all( a.p == b.p )
    np.all( a.r == b.r )
""").split("\n")))

if __name__ == '__main__':
    a_fold = os.path.expandvars("/tmp/$USER/opticks/QSimTest/mock_propagate")
    b_fold = os.path.expandvars("/tmp/$USER/opticks/QSimTest/mock_propagate_2")
    a = Fold.Load(a_fold)
    b = Fold.Load(b_fold)

    for e in expr:
        v = eval(e)
        print("eval(%s) = %s " % (e,v))
        assert v 
    pass







