#!/usr/bin/env python
"""
sdEllipse.py
===============

Plots (-1mm, +1mm) distance to ellipse contours together with local frame PMT hit positions. 
This allows verifying exactly which ellipsoid hits are being collected at. 
The ellipse contour scans are created by sdEllipse.cc

"""

import numpy as np
import matplotlib.pyplot as plt

from j.PMTEfficiencyCheck_ import PMTEfficiencyCheck_
pec = PMTEfficiencyCheck_()


label = ["body ellipse", "inner1 ellipse"]
elli = ["254_190", "249_185"] 
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

rz0 = pec.rz()

#zoom = 0
zoom = 30


for i in range(len(elli)):

    g = np.load("/tmp/sdEllipse_grid_%s.npy" % elli[i])
   
    #pp = np.load("/tmp/sdEllipse_points_%s.npy" % elli[i])
    pp = None

    x = np.unique(g[:,0]) 
    y = np.unique(g[:,1])
    Z = g[:,2].reshape(len(x),len(y))  

    X, Y = np.meshgrid(x, y) 

    ax = axs[i] 
    ax.contour(X, Y, Z, colors='black', levels=[-1,1] )
    ax.set_aspect('equal')

    ax.set_title("%s : %s " % (label[i],  elli[i]))

    if zoom > 0: 
        ax.set_xlim( rz0[0]-zoom, rz0[0]+zoom)  
        ax.set_ylim( rz0[1]-zoom, rz0[1]+zoom)  
    pass

    if not pp is None:
        ax.scatter( pp[:,0], pp[:,1] )
        for p in pp:
            ax.text(p[0], p[1], p[2])
        pass
    pass

pass


sli = slice(0,10000)
pec.rz_plot(axs, 0, sli )


fig.show() 

