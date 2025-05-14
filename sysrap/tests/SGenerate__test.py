#!/usr/bin/env python
"""
SGenerate__test.py
====================

1. load f from $FOLD
2. check f.gs f.ph shapes
3. for MODE=3 make 3D plot of generated photons

::

    hookup_conda_ok  # BYOC
    MODE=3 ~/o/sysrap/tests/SGenerate__test.sh pdb


"""
import os, numpy as np
from opticks.ana.fold import Fold

MODE = int(os.environ.get("MODE",0))

if MODE in [2,3]:
    import opticks.ana.pvplt as pvp
else:
    pvp = None
pass

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))


    GS_NAME = os.environ.get("SGenerate__test_GS_NAME", "gs.npy").replace(".npy", "")
    PH_NAME = os.environ.get("SGenerate__test_PH_NAME", "ph.npy").replace(".npy", "")

    gs = getattr(f, GS_NAME)
    ph = getattr(f, PH_NAME)

    print("GS_NAME %s gs.shape %s\n" % (GS_NAME, str(gs.shape)))
    print("PH_NAME %s ph.shape %s\n" % (PH_NAME, str(ph.shape)))

    pos = ph[:,0,:3]
    mom = ph[:,1,:3]
    pol = ph[:,2,:3]

    if MODE == 3:
        pl = pvp.pvplt_plotter()
        #pl.add_points(pos, point_size=20)
        #pvplt_arrows(pl, pos, mom, factor=20 )

        pvp.pvplt_polarized(pl, pos, mom, pol, factor=20  )

        pl.show()
    pass
