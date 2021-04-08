#!/usr/bin/env python
"""




"""
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp

# non-monotonic 
xd = np.array([0., 1., 1., 2.])
yd = np.array([1., 2., 0., 1.]) 

inp = {}

kinds = "linear nearest previous next".split()
kinds += "zero slinear".split()
#kinds += ["cubic"]
#kinds += ["quadratic"]

for kind in kinds:
    inp[kind] = sp.interpolate.interp1d(xd, yd, kind=kind )
pass

fig, axs = plt.subplots(1)
ax = axs

x = np.linspace(0, 2, 500)
for kind in kinds:
    ax.plot(x, inp[kind](x), label=kind )
pass

plt.legend()
fig.show()


