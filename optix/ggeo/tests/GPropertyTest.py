#!/usr/bin/env python
"""
Usage::

    ./GPropertyTest.sh


http://stackoverflow.com/questions/9497524/displaying-3-histograms-on-1-axis-in-a-legible-way-matplotlib

"""
import numpy as np
import matplotlib.pyplot as plt


psample = 1/np.load("/tmp/psample.npy") 
isample = 1/np.load("/tmp/isample.npy") 
#isample = np.load("/tmp/isample.npy") 


params = dict(bins=100, 
              range=(0, 900), 
              normed=False, 
              log=True, histtype='step')

plt.hist(psample, label='p', **params)
plt.hist(isample, label='i', **params)
plt.show()


