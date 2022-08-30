#!/usr/bin/env python
"""
cf_G4CXSimtraceTest.py
==============================

"""

import os, numpy as np, logging
log = logging.getLogger(__name__)
from opticks.ana.fold import Fold
from opticks.CSG.Values import Values 
import matplotlib.pyplot as mp

SIZE = np.array([1280, 720]) 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    t = Fold.Load("$T_FOLD", symbol="t")
    s = Fold.Load("$S_FOLD", symbol="s")

    print(repr(t))
    print(repr(s))

    tv = Values.Find("$T_FOLD", symbol="tv")
    sv = Values.Find("$S_FOLD", symbol="sv")
 
    print(repr(tv))
    print(repr(sv))


    t_hit = t.simtrace[:,0,3]>0
    s_hit = s.simtrace[:,0,3]>0

    t_pos = t.simtrace[t_hit][:,1,:3]
    s_pos = s.simtrace[s_hit][:,1,:3]


    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.set_aspect('equal')
    ax.scatter( t_pos[:,0], t_pos[:,2], label="t_pos", s=1 ) 
    ax.scatter( s_pos[:,0], s_pos[:,2], label="s_pos", s=1 ) 

    fig.show()

