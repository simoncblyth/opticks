#!/usr/bin/env python

import os, logging, builtins, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.fold import Fold 

import matplotlib.pyplot as mp
from opticks.ana.fold import Fold
from opticks.sysrap.sframe import sframe , X, Y, Z

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    s = Fold.Load(symbol="s")
    print(repr(s))
    fr = s.sframe
    s_geom = os.environ["GEOM"]

    fig, ax = fr.mp_subplots(mp)  
    
    if not s is None:
        s_hit = s.simtrace[:,0,3]>0 
        s_pos = s.simtrace[s_hit][:,1,:3]
    pass

    if not s is None:
        fr.mp_scatter(s_pos, label="%s" % s_geom, s=1 )
    pass

    ax.legend()
    fig.show()
pass


