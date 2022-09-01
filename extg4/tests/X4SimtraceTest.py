#!/usr/bin/env python
"""
X4SimtraceTest.py
===================

"""
import os, logging, builtins, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as mp
from opticks.ana.fold import Fold
from opticks.sysrap.sframe import sframe , X, Y, Z

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    ff = Fold.MultiLoad()

    frs = list(filter(None, map(lambda f:f.sframe, ff)))
    assert len(frs) > 0
    fr = frs[0]       ## HMM: picking first frame, maybe need to form composite bbox from all frames ?


    s_hit = s.simtrace[:,0,3]>0 
    s_pos = s.simtrace[s_hit][:,1,:3]

    t_hit = t.simtrace[:,0,3]>0 
    t_pos = t.simtrace[t_hit][:,1,:3]


    fig, ax = fr.mp_subplots(mp)  

    fr.mp_scatter(s_pos, label="%s" % s_geom, s=1 )
    fr.mp_scatter(t_pos, label="%s" % t_geom, s=1 )

    ax.legend()
    fig.show()



