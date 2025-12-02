#!/usr/bin/env python
"""
QEvtTest_PerLaunchMerge.py
============================

~/o/qudarap/tests/QEvtTest.sh pdb1
~/o/qudarap/tests/QEvtTest.sh info_run_pdb1


"""

import os, numpy as np
TEST = os.environ["TEST"]

from opticks.ana.fold import Fold, EVAL
from opticks.sysrap.sphoton import SPhoton
from opticks.sysrap.sphotonlite import SPhotonLite


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    pf = SPhoton.view(f.photon)
    hm = SPhoton.view(f.hitmerged)

    pl = SPhotonLite.view(f.photonlite)
    hlm = SPhotonLite.view(f.hitlitemerged)

    EVAL(r"""
    TEST

    pf.shape
    pl.shape
    hm.shape
    hlm.shape

    # compare with expectations of merge result
    np.all( f.hitmerged == f.x_hitmerged )
    np.all( f.hitlitemerged == f.x_hitlitemerged )

    # lite or full should make no difference
    np.all( hm.identity == hlm.identity )
    np.all( hm.time == hlm.time )

    """, ns=locals())



