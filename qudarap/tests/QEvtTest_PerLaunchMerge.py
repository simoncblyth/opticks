#!/usr/bin/env python
"""
QEvtTest_PerLaunchMerge.py
============================


"""

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sphoton import SPhoton
from opticks.sysrap.sphotonlite import SPhotonLite


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    p = SPhoton.view(f.photon)
    h = SPhoton.view(f.hitmerged)
    print("p,h")

    l = SPhotonLite.view(f.photonlite)
    i = SPhotonLite.view(f.hitlitemerged)
    print("l,i")






