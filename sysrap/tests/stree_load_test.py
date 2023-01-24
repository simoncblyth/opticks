#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree

if __name__ == '__main__':

    f = Fold.Load(symbol="f")
    print(repr(f))

    st = stree(f)
    print(repr(st))

    u_bd, n_bd = np.unique( st.nds.boundary, return_counts=True ) 
    for i in range(len(u_bd)):
        u = u_bd[i]
        n = n_bd[i]
        print(" %3d : %4d : %6d : %s " % (i, u, n, st.f.bd_names[u] )) 
    pass

    print(st.desc_boundary())




