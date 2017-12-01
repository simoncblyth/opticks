#!/usr/bin/env python
"""
NRngDiffuseTest.py 
=======================

See npy-/tests/NRngDiffuseTest.cc

"""
import os, sys, logging
import numpy as np
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(suppress=True, precision=3)


if __name__ == '__main__':


    plt.ion()
    fig = plt.figure()


    
    d = np.load(os.path.expandvars("$TMP/NRngDiffuseTest_diffuse.npy"))
    s = np.load(os.path.expandvars("$TMP/NRngDiffuseTest_sphere.npy"))


    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(d[:,0], d[:,1], d[:,2])

     

