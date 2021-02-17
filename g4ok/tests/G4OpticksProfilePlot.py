#!/usr/bin/env python

import os, sys
import numpy as np
from numpy.polynomial import Polynomial as Poly

try:
    import matplotlib.pyplot as plt
except ImportError:
     plt = None
pass

#plt = None  


path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OPTICKS_PROFILE_PATH") 


path = os.path.expandvars(path)
a = np.load(path)
print(a)

delta = (a[-1,1] - a[0,1])
slope0 = delta/len(a)
msg = " delta:%10.2f slope0:%10.2f " % ( delta, slope0 )
print(msg)


x = np.arange(len(a))
y = a[:,1]   # VM 

#p = Poly.fit(x,y,1, domain=(x.min(), x.max()), window=(y.min(),y.max()) )  
# coeffs are somehow transformed from the obvious slope/intercept with Poly ?

p = np.poly1d(np.polyfit(x,y,1))  ## old way gives expected coef meanings immediately  


label = "line fit:  slope %10.2f    intercept %10.2f " % (p.coef[0], p.coef[1])
print(label)


if plt:
    plt.ion()
    plt.close()

    fig, ax = plt.subplots(1, 1)

    ax.plot( x, y , 'o' )
    ax.plot(x, p(x), lw=2)
    plt.xlabel(msg + " " + label)

    plt.show()
pass

