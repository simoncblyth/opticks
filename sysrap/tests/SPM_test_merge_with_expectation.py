#!/usr/bin/env python
"""
SPM_test_merge_with_expectation.py
====================================

  : t.NPFold_index                                     :                 (9,) : 0:00:48.073502
  : t.NPFold_names                                     :                 (0,) : 0:00:48.073502
  : t.x_merged                                         :         (1000, 4, 4) : 3:17:03.900429
  : t.merged                                           :         (1000, 4, 4) : 3:17:03.761429


  : t.photon0                                          :       (500500, 4, 4) : 0:00:48.412502
  : t.photon                                           :       (500500, 4, 4) : 0:00:48.284502

  : t.photonlite0                                      :          (500500, 4) : 0:00:48.151502
  : t.photonlite                                       :          (500500, 4) : 0:00:48.132502

  : t.x_hitmerged                                      :         (1000, 4, 4) : 0:00:48.415502
  : t.hitmerged                                        :         (1000, 4, 4) : 0:00:48.153502

  : t.x_hitlitemerged                                  :            (1000, 4) : 0:00:48.153502
  : t.hitlitemerged                                    :            (1000, 4) : 0:00:48.101502

  : t.c_hitlitemerged                                  :            (1000, 4) : 0:00:48.073502





"""

import numpy as np
np.set_printoptions(threshold=50, edgeitems=100, linewidth=120)



from opticks.ana.fold import Fold
from opticks.sysrap.sphoton     import SPhoton
from opticks.sysrap.sphotonlite import SPhotonLite, SPhotonLite_Merge, EFFICIENCY_COLLECT

if __name__ == '__main__':
    t = Fold.Load("$TFOLD",symbol="t")
    print(repr(t))



    p = SPhoton.view(t.photon)
    print("p\n", repr(p))

    x_hm = SPhoton.view(t.x_hitmerged)
    print("x_hm\n", repr(x_hm))

    hm = SPhoton.view(t.hitmerged)
    print("hm\n", repr(hm))


    # merging just combines, so not very floating point sensitive : expect exact match
    assert np.all( t.hitmerged == t.x_hitmerged )



    pl = SPhotonLite.view(t.photonlite)
    print("pl\n", repr(pl))

    x_hlm = SPhotonLite.view(t.x_hitlitemerged)
    print("x_hlm\n", repr(x_hlm))

    hlm = SPhotonLite.view(t.hitlitemerged)
    print("hlm\n", repr(hlm))

    c_hlm = SPhotonLite.view(t.c_hitlitemerged)
    print("c_hlm\n", repr(c_hlm))


    # merging just combines, so not very floating point sensitive : expect exact match
    assert np.all( t.hitlitemerged == t.x_hitlitemerged )

    assert np.all( t.hitlitemerged == t.c_hitlitemerged )  # get match after avoid f_dummy zero



    assert np.all( hlm.hitcount == c_hlm.hitcount )
    assert np.all( hlm.identity == c_hlm.identity )
    assert np.all( hlm.time == c_hlm.time )
    assert np.all( hlm.flagmask == c_hlm.flagmask )
    assert np.all( hlm.lposcost == c_hlm.lposcost )
    assert np.all( hlm.lposfphi == c_hlm.lposfphi )        # required avoiding pos [-1,0,0] phi/fphi discontinuity




