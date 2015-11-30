#!/usr/bin/env python
"""

* http://www.yale.edu/ceo/Documentation/ComputingThePlanckFunction.pdf

::


             2hc^2 w^-5
   Bw(T) =   ----------------
              e^(beta) - 1


             hc/kT
    beta =   -----
              w



"""

import matplotlib.pyplot as plt
import numpy as np

def planck(nm, K):
    #from scipy.constants import h,c,k
    h = 6.62606957e-34
    c = 299792458.0
    k = 1.3806488e-23

    wav = nm/1e9 
    a = 2.0*h*c*c
    b = (h*c)/(k*K)/wav
    return a/(np.power(wav,5) * np.expm1(b))  

w = np.arange(100,1000,1)

for temp in np.arange(4000,10000,500):
    intensity = planck(w, temp)
    plt.plot(w, intensity, '-') 




plt.ion()
plt.show()




