#!/usr/bin/env python

import sys, numpy as np
from opticks.ana.fold import Fold

from opticks.ana.pvplt import *

MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,-2,3,-3]

if __name__ == '__main__':
    t = Fold.Load(symbol="t")  # from $FOLD
    print(repr(t))
    match = np.all( t.fast == t.slow )
    rc = 0 if match else 1
    print(f" fast/slow match {match} rc {rc}")

    icdf_0 = t.icdf[0].reshape(-1)
    prob = np.linspace(0,1,len(icdf_0))

    if MODE == 2:
        label = "icdf_0"
        fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label)
        ax = axs[0]
        ax.set_aspect('auto')
        ax.set_ylim(icdf_0.min(),icdf_0.max())
        ax.set_xlim(0,1)
        ax.scatter( prob, icdf_0, s=1, c="b" )
        fig.show()
    pass

    sys.exit(rc)


