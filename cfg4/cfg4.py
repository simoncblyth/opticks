#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

from env.numerics.npy.ana import Evt, Selection, costheta_, cross_
deg = np.pi/180.


if __name__ == '__main__':
    s_evt = Evt(tag="-5", det="rainbow", label="G4 S")
    p_evt = Evt(tag="-6", det="rainbow", label="G4 P")

    s_evt.history_table()
    p_evt.history_table()


