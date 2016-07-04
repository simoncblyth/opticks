#!/usr/bin/env python
"""
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from opticks.ana.ana import Evt, Selection, costheta_, cross_
from opticks.ana.geometry import Boundary   
deg = np.pi/180.


if __name__ == '__main__':
    pass

    spol, ppol = "5", "6"
    g = Evt(tag="-"+spol, det="rainbow", label="S G4")
    o = Evt(tag=spol, det="rainbow", label="S Op")

    # check magnitude of polarization
    for e in [g,o]: 
        mag = np.linalg.norm(e.rpol_(0),2,1)
        assert mag.max() < 1.01 and mag.min() > 0.99



