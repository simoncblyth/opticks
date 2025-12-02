#!/usr/bin/env python
"""
SPM_test_merge_with_expectation.py
====================================



"""

import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sphoton     import SPhoton
from opticks.sysrap.sphotonlite import SPhotonLite, SPhotonLite_Merge, EFFICIENCY_COLLECT

if __name__ == '__main__':
    t = Fold.Load("$TFOLD",symbol="t")
    print(repr(t))

    x = SPhoton.view(t.x_merged)
    print("x\n", repr(x))

    m = SPhoton.view(t.merged)
    print("m\n", repr(m))

    p = SPhoton.view(t.photon)
    print("p\n", repr(p))


    assert np.all( t.hitmerged == t.x_hitmerged )

    assert np.all( t.hitlitemerged == t.x_hitlitemerged )

    assert np.all( t.hitlitemerged == t.c_hitlitemerged )




