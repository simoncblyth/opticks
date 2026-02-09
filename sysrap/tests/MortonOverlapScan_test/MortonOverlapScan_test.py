#!/usr/bin/env python
"""


In [4]: f.simtrace_overlap.view(np.uint32)[:,2,3] >> 16
Out[4]: array([3255, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844, 2844], dtype=uint32)

In [5]: prim = f.simtrace_overlap.view(np.uint32)[:,2,3] >> 16

In [6]: np.unique(prim, return_counts=True)
Out[6]: (array([2844, 3255], dtype=uint32), array([26,  1]))

In [7]: np.c_[np.unique(prim, return_counts=True)]
Out[7]:
array([[2844,   26],
       [3255,    1]])



"""

import os
import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load("$MFOLD", symbol="f")
    print(repr(f))

    print(f" f.sfr.meta.name    {f.sfr.meta.name} ")
    print(f" f.sfr.meta.treedir {f.sfr.meta.treedir} ")

    prn = np.genfromtxt(os.path.join(f.sfr.meta.treedir,"prname_names.txt"),dtype=str, delimiter="\n")

    prim_bnd = f.simtrace_overlap.view(np.uint32)[:,2,3]
    prim = prim_bnd >> 16

    u_prim, n_prim = np.unique(prim, return_counts=True)

    tab = np.c_[u_prim,n_prim,prn[u_prim]]
    print(prim)
    print(tab)
pass

