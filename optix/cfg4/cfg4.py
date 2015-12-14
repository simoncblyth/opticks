#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

from env.numerics.npy.ana import Evt, Selection, costheta_, cross_
deg = np.pi/180.


if __name__ == '__main__':
    evt = Evt(tag="-1", det="rainbow", label="G4")
    evt.history_table()


