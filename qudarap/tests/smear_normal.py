#!/usr/bin/env python
"""
smear_normal.py
===============

"""

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
import matplotlib.pyplot as mp
SIZE = np.array([1280, 720])

MODE=int(os.environ.get("MODE","3")) 
TEST = os.environ.get("TEST", "")
CHECK = os.environ.get("CHECK", "SmearNormal_SigmaAlpha")

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    is_mock = not getattr(f, CHECK, None ) is None
    if not is_mock: CHECK = "q"

    q = getattr(f, CHECK)
    assert len(q.shape) == 2
    assert q.shape[1] == 4

    a = q[:,:3]

    m = getattr(f, "%s_meta" % CHECK ) 

    source = m.source[0] if hasattr(m,'source') else "NO-source-metadata" 
    value = m.value[0] if hasattr(m,'value') else None
    valuename = m.valuename[0] if hasattr(m,'valuename') else None

    label = "%s " % source 
    label += " white %s:%s " % ( valuename, value )
    

    if MODE == 3:
        pl = pvplt_plotter(label=label)
        pvplt_viewpoint( pl )

        pos = np.array( [[0,0,0]] )
        vec = np.array( [[0,0,1]] ) 
        pvplt_lines( pl, pos, vec )

        pl.add_points( a , color="white" )

        cpos = pl.show()

    elif MODE == 2:
         
        fig, ax = mp.subplots(figsize=SIZE/100.)
        fig.suptitle(label)
        ax.set_aspect("equal")

        ax.scatter( a[:,0], a[:,1], s=0.1, c="b" )
        fig.show()
 
    elif MODE == 1:

        nrm = np.array( [0,0,1], dtype=np.float32 )  ## unsmeared normal is +Z direction  
        ## dot product with Z direction picks Z coordinate 
        ## so np.arccos should be the final alpha 

        angle = np.arccos(np.dot( a, nrm )) 

        bins = np.linspace(0,0.4,100)  
        h_angle = np.histogram(angle, bins=bins )[0] 

        fig, ax = mp.subplots(figsize=SIZE/100.)
        fig.suptitle(label)

        if not h_angle is None:ax.plot( bins[:-1], h_angle,  drawstyle="steps-post", label="h_angle" )
        ax.legend() 

        fig.show()
    pass
pass


