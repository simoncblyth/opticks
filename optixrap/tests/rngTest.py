#!/usr/bin/env python
"""
rngTest.py
==================

::

    ipython -i tests/rngTest.py 

"""
import os, sys, logging, numpy as np
import matplotlib.pyplot as plt
from opticks.ana.nload import np_load
from opticks.ana.key import keydir

log = logging.getLogger(__name__)

if __name__ == '__main__':
    path = os.path.expandvars("$TMP/optixrap/rngTest/out.npy") 
    u = np.load(path)

    title = " rngTest.py u:%s"%str(u.shape)
    print(title)

    plt.ion()

    ubin = np.linspace(0,1, 20)   

    counts, edges = np.histogram(u, bins=ubin )
    fcounts = counts.astype(np.float32)
    fcounts  /= fcounts.sum()

    plt.close()

    plt.title(title)

    plt.plot( edges[:-1], fcounts, drawstyle="steps-mid")

    plt.axis( [0, 1, 0, fcounts.max()*2 ] )


