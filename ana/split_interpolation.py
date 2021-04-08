#!/usr/bin/env python
"""

* https://stackoverflow.com/questions/48028766/get-x-values-corresponding-to-y-value-on-non-monotonic-curves

"""

import numpy as np
from opticks.ana.mlib import GMaterialLib
mlib = GMaterialLib()

import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d

class SplitInterpolation(object):
    def __init__(self, x, y ):l




if __name__ == '__main__':
   
    #name = "LS"
    name = "Water"
    ri = mlib("%s.RINDEX" % name).copy()

    # energy(eV) and refractive index in ascending energy 

    #x,y = mlib.ev[::-1], ri[::-1]
    #x,y = mlib.nm, ri 
    x,y = ri, mlib.nm,


    order = np.argsort(x)
    xs, ys = x[order], y[order]

    fig, axs = plt.subplots(2)
    axs[0].plot( x, y )
    axs[1].plot(xs, ys )
    fig.show()


    # compute indices of points where y changes direction
    ydir = np.sign(np.diff(ys))
    yturn = 1 + np.where(np.diff(ydir) != 0)[0]

    # find groups of x and y within which y is monotonic
    xgrp = np.split(xs, yturn)
    ygrp = np.split(ys, yturn)


if 0:
    interps = [interp1d(y, x, bounds_error=False) for y, x in zip(ygrp, xgrp)]

    # interpolate all y values
    yval = 100
    xvals = np.array([interp(yval) for interp in interps])

    print(xvals)
    # array([          nan,  122.96996037,  207.62395521,           nan])





