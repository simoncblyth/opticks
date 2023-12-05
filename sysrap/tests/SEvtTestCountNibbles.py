#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))
   
    tab0 = f.seqnib_table
    u_seqnib, n_seqnib = np.unique( f.seqnib, return_counts=True )
    print(np.c_[u_seqnib, n_seqnib])

    assert np.all( tab0[2:] == n_seqnib  )  

