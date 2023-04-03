#!/usr/bin/env python


import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *

X,Y,Z = 0,1,2
H,V = X,Z

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    print("MODE:%d" % MODE)

    pl = plotter(label="U4SimtracePlot.py")  # MODE:2 (fig,ax)  MODE:3 pv plotter

    if MODE == 2:
        fig, axs = pl
        assert len(axs) == 1
        ax = axs[0]
    elif MODE == 3:
        pass
    pass

    color = "red"
    label = "klop"
    SZ = 1
 

    inrm = t.simtrace[:,0].copy()  # normal at intersect (not implemented)
    inrm[:,3] = 0

    lpos = t.simtrace[:,1].copy()  # intersect position 
    lpos[:,3] = 1

    tpos = t.simtrace[:,2].copy()  # trace origin
    tpos[:,3] = 1 

    tdir = t.simtrace[:,3].copy()  # trace direction
    tdir[:,3] = 0 



    if MODE == 2:
        ax.scatter( lpos[:,H], lpos[:,V], s=SZ, color=color, label=label )
        ax.scatter( tpos[:,H], tpos[:,V], s=SZ, color=color, label=label )
    elif MODE == 3:
        pl.add_points( gpos[:,:3], color=color, label=label)
    pass


    if MODE == 2:
        fig.show()
    elif MODE == 3:
        pl.show()
    pass

pass


