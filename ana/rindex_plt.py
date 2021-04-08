#!/usr/bin/env python
"""

https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html


"""
import math 
import numpy as np
from scipy.interpolate import interp1d 
from scipy.interpolate import CubicSpline
from opticks.ana.mlib import GMaterialLib
mlib = GMaterialLib()

import matplotlib.pyplot as plt 

if __name__ == '__main__':

    names = []
    for name in mlib.names:
        a = mlib("%s.RINDEX" % name )
        one_value = np.all(a == a[0])
        print("name:%-20s mn:%10.4f mx:%10.4f  %s " % (name, a.min(), a.max(), "one-value" if one_value else "-"))
        if one_value: continue
        pass 
        names.append(name)
    pass 
    print(names)
    names = ["LS"]

    fig, axs = plt.subplots(len(names), sharex=True)
    if len(names) == 1: axs = [axs]
    for i in range(len(names)):
        ax = axs[i]
        name = names[i]
        ri = mlib("%s.RINDEX" % name) 

        #kind = "cubic"
        #kind = "quadratic"
        #kind = "linear"
        #interp = interp1d( mlib.nm,  ri, kind=kind ) 

        interp = CubicSpline( mlib.nm, ri )

        ax.plot(mlib.nm, ri)

        nm_10 = np.linspace( mlib.nm[0], mlib.nm[-1], len(mlib.nm)*10 ) 
        ri_10 = interp(nm_10) 

        #i_interp = interp1d( ri, mlib.nm, kind=kind ) 
        #i_nm_10 = i_interp(ri_10)
        #ax.scatter(i_nm_10, ri_10, s=4.0)

        ax.scatter(nm_10, ri_10, s=4.0)
        ax.text(1,1, name, ha='right', va='top', transform=ax.transAxes ) 
        ax.scatter(mlib.nm, ri, s=8.0, c='r')
    pass
    fig.show()




