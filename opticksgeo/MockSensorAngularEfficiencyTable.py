#!/usr/bin/env python
"""

::

    ipython -i MockSensorAngularEfficiencyTable.py

"""
import os, numpy as np
path = os.path.expandvars("/tmp/$USER/opticks/opticksgeo/tests/MockSensorAngularEfficiencyTableTest.npy")

a = np.load(path)
assert len(a.shape) == 3 

ctx = dict(name=os.path.basename(path),shape=a.shape,num_cat=a.shape[0],num_theta=a.shape[1],num_phi=a.shape[2])
title = "{name} {shape!r} num_theta:{num_theta} num_phi:{num_phi}".format(**ctx)
print(title)

try:
    import matplotlib.pyplot as plt 
except ImportError:
    plt = None
pass

if plt:
    fig, axs = plt.subplots(ctx["num_cat"])
    fig.suptitle(title)
    for i in range(ctx["num_cat"]):
        ax = axs[i] if ctx["num_cat"] > 1 else axs
        ax.imshow(a[i])
    pass
    plt.ion()
    plt.show()    
pass

