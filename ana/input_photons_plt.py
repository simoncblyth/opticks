#!/usr/bin/env python
"""

"""
import os, sys, numpy as np
from opticks.ana.input_photons import InputPhotons
from opticks.ana.pvplt import *


if __name__ == '__main__':

    path = InputPhotons.BasePath()  # sensitive to OPTICKS_INPUT_PHOTON eg RandomSpherical100_f4.npy
    p = np.load(path) 

    pl = pvplt_plotter(label=path)

    pos = p[:,0,:3]
    mom = p[:,1,:3]
    pol = p[:,2,:3] 

    pvplt_polarized( pl, pos, mom, pol, factor=10 )

    cp = pl.show()
pass

