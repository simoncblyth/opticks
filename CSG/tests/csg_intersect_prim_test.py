#!/usr/bin/env python
"""
csg_intersect_prim_test.py
===========================

   [:,0,3] > 0





"""

import os, logging, numpy as np
from opticks.ana.fold import Fold, IsRemoteSession
from opticks.ana.pvplt import pvplt_show
MODE = int(os.environ.get("MODE","3"))

log = logging.getLogger(__name__)


if IsRemoteSession():  # HMM: maybe do this inside pvplt ?
    MODE = 0
    print("detect fold.IsRemoteSession forcing MODE:%d" % MODE)
elif MODE in [2,3]:
    from opticks.ana.pvplt import *  # HMM this import overrides MODE, so need to keep defaults the same
pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    f = Fold.Load("$FOLD/$TEST", symbol="f")
    print(repr(f))
    xy = f.xy

    H,V = 0,1  # X, Y
    sc = 10

    for e in [xy]:

        nrm = e[:,0,:3]
        r_nrm = np.sqrt(np.sum( nrm*nrm, axis=1 ))
        print(r_nrm)


        label = "CSG/tests/csg_intersect_prim_test.sh "

        if MODE in [0,1]:
            print("not plotting as MODE %d in environ" % MODE )
        elif MODE == 2:
            pl = mpplt_plotter(label=label)
            fig, axs = pl
            assert len(axs) == 1
            ax = axs[0]
            ax.set_xlim(-120,120)

        elif MODE == 3:
            pl = pvplt_plotter(label)
            pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars


        pass


        spos = e[:,1,:3]   # origin position
        snrm = e[:,0,:3]


        if MODE == 2:
            ax.scatter( spos[:,H], spos[:,V], s=0.1 )

            for i in range(len(spos)):
                p = spos[i]
                n = snrm[i]
                ax.arrow( p[H], p[V], sc*n[H], sc*n[V] )
            pass

        elif MODE == 3:
            pl.add_points(spos[:,:3])
        else:
            pass
        pass

        if MODE == 2:
            fig.show()
        elif MODE == 3:
            pvplt_show(pl)
        pass
    pass
pass
