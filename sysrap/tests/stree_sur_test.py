#!/usr/bin/env python

import numpy as np, textwrap
from opticks.ana.fold import Fold
import matplotlib.pyplot as plt
SIZE = np.array([1280,720])

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    s = Fold.Load("$FOLD/stree", symbol="s")
    print(repr(s))

    osur = s.implicit_osur.reshape(-1,8)
    isur = s.implicit_isur.reshape(-1,8)

    mtn = np.array(s.mat_names)
    sun = np.array(s.sur_names)
    

    u_osur,i_osur,n_osur = np.unique(osur,axis=0,return_index=True,return_counts=True)
    u_isur,i_isur,n_isur = np.unique(isur,axis=0,return_index=True,return_counts=True)


    EXPRS = r"""

    osur.shape
    u_osur.shape

    isur.shape
    u_isur.shape

    np.c_[n_osur,i_osur,u_osur]
    np.c_[n_isur,i_isur,u_isur]

    """
    for expr in list(filter(None,textwrap.dedent(EXPRS).split("\n"))):
        print(expr)
        if expr[0] in " #": continue
        print(eval(expr))
    pass



