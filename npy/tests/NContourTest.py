#!/usr/bin/env python
"""

./NContourTest.sh 

"""

import os, numpy as np

from opticks.ana.fold import Fold
import matplotlib.pyplot as mp

if __name__ == '__main__':

    name = "NContourTest.py"
    geom = os.environ.get("GEOM", "Cone_0" ) 
    original = Fold.Load("/tmp/$USER/opticks/npy/NContourTest", geom, "original" )
    modified = Fold.Load("/tmp/$USER/opticks/npy/NContourTest", geom, "modified" )
    title = [name, original.base, modified.base  ]
        
    mp.ion()
    mp.close()

    levels = np.array([0.,])

if 1:
    fig, ax = mp.subplots()
    fig.suptitle("\n".join(title))

    qcs0 = ax.contour(original.X, original.Y, original.Z, levels )
    mp.clabel(qcs0, inline=1, fontsize=10)

    qcs1 = ax.contour(modified.X, modified.Y, modified.Z, levels )
    mp.clabel(qcs1, inline=1, fontsize=10)

    mp.show()





if 0:
    fig, axs = mp.subplots(2)
    ax = axs[0]
    ax = axs[1]


