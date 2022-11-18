#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

from opticks.ana.pvplt import * 


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    fs = t.SFastSim_Debug   


    pos = fs[:,0,:3]

    pl = pvplt_plotter("U4PMTFastSimTest.py:SFastSim_Debug" )
    pl.add_points( pos, color="white" )

    pl.show()


pass
