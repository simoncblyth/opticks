#!/usr/bin/env python
"""
random_direction_marsaglia.py
==========================================
::

   EYE=-1,-1,1 LOOK=0,0,0.5 PARA=1 ./QSimTest.sh ana


"""
import os, numpy as np
from opticks.ana.fold import Fold
MODE = int(os.environ.get("MODE","0"))

if MODE in [2,3]:
    from opticks.ana.pvplt import *
    import pyvista as pv
pass


TEST = os.environ["TEST"]
GUI = not "NOGUI" in os.environ


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    q = t.q

    lim = slice(0,10000)

    print( " TEST : %s " % TEST)
    print( "q.shape %s " % str(q.shape) )


    if MODE == 3:
        print(" using lim for plotting %s " % lim )
        label = TEST
        pl = pvplt_plotter(label=label)   

      
        pvplt_viewpoint( pl ) 
        pl.add_points( q[:,:3][lim] )


        outpath = os.path.expandvars("$FOLD/figs/%s.png" % label )
        outdir = os.path.dirname(outpath)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        pass

        print(" outpath: %s " % outpath ) 
        cp = pl.show(screenshot=outpath) if GUI else None
    pass
pass       
