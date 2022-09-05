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


    SYMBOLS = os.environ.get("SYMBOLS", None)
    FOLD = os.environ.get("FOLD", None)

    if not FOLD is None:
        s = Fold.Load(symbol="s")
        t = None
        fr = s.sframe
        s_geom = os.environ["GEOM"]
    elif not SYMBOLS is None:
        ff = Fold.MultiLoad()
        frs = list(filter(None, map(lambda f:f.sframe, ff)))
        assert len(frs) > 0
        fr = frs[0]       ## HMM: picking first frame, maybe need to form composite bbox from all frames ?
    else:
        assert 0 
    pass

    fig, ax = fr.mp_subplots(mp)  

    if not s is None:
        s_hit = s.simtrace[:,0,3]>0 
        s_pos = s.simtrace[s_hit][:,1,:3]
    pass

    if not t is None:
        t_hit = t.simtrace[:,0,3]>0 
        t_pos = t.simtrace[t_hit][:,1,:3]
    pass

    if not s is None:
        fr.mp_scatter(s_pos, label="%s" % s_geom, s=1 )
    pass

    if not t is None:
        fr.mp_scatter(t_pos, label="%s" % t_geom, s=1 )
    pass

    ax.legend()
    fig.show()



