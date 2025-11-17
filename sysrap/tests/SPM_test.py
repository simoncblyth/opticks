#!/usr/bin/env python
"""
SPM_test.py
============



"""

import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sphotonlite import SPhotonLite, SPhotonLite_Merge, EFFICIENCY_COLLECT

if __name__ == '__main__':

    f = Fold.Load("$AFOLD",symbol="f")

    _pl = f.photonlite
    _hlm = f.hitlitemerged   # from CUDA/Thrust impl

    m = SPhotonLite_Merge(tw = 1.0, select_mask = EFFICIENCY_COLLECT)
    _hlm2 = m(_pl)

    pl = SPhotonLite.view(_pl)
    hlm = SPhotonLite.view(_hlm)
    print("hlm[:10]\n",repr(hlm[:10]))

    hlm2 = SPhotonLite.view(_hlm2)
    print("hlm2[:10]\n",repr(hlm2[:10]))


    assert np.all( hlm.identity == hlm2.identity )
    assert np.all( hlm.time == hlm2.time )
    assert np.all( hlm.hitcount == hlm2.hitcount )
    assert np.all( hlm.lposfphi == hlm2.lposfphi )
    assert np.all( hlm.lposcost == hlm2.lposcost )
    assert np.all( hlm.flagmask == hlm2.flagmask )

