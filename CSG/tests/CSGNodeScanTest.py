#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.fold import Fold

import matplotlib.pyplot as mp
from opticks.sysrap.sframe import sframe , X, Y, Z
from opticks.ana.pvplt import mpplt_simtrace_selection_line, mpplt_hist



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    s = Fold.Load(symbol="s")
    print(repr(s))

    s_simtrace = s.simtrace.reshape(-1,4,4)
    s_geom = os.environ.get("GEOM", "geom")

    fr = sframe.FakeXZ(e=11) 

    fig, ax = fr.mp_subplots(mp)  

    if not s is None:
        s_hit = s_simtrace[:,0,3]>0 
        s_pos = s_simtrace[s_hit][:,1,:3]
        fr.mp_scatter(s_pos, label="%s" % s_geom, s=1 )
    pass
 
    if not "NOLEGEND" in os.environ:
        ax.legend()
    pass
    fig.show()

