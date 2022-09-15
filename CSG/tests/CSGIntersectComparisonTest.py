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

    ab = s.a_simtrace - s.b_simtrace 
    print(ab.max())

    a_simtrace = s.a_simtrace.reshape(-1,4,4)
    b_simtrace = s.b_simtrace.reshape(-1,4,4)

    fr = sframe.FakeXZ(e=300) 

    fig, ax = fr.mp_subplots(mp)  

    if not s is None:
        a_hit = a_simtrace[:,0,3]>0 
        a_pos = a_simtrace[a_hit][:,1,:3]
        fr.mp_scatter(a_pos, label="a_pos", s=1 )

        b_hit = b_simtrace[:,0,3]>0 
        b_pos = b_simtrace[b_hit][:,1,:3]
        fr.mp_scatter(b_pos, label="b_pos", s=1 )

        #w = np.logical_and( np.abs(s.b_simtrace[:,1,0]) < 10. , np.abs(s.b_simtrace[:,1,2]) < 10. )  
    pass
 
    if not "NOLEGEND" in os.environ:
        ax.legend()
    pass
    fig.show()
pass

