#!/usr/bin/env python
"""
aoi.py
=========

* see ~/j/Layr/Layr.h 

HMM: more physical to use dot(photon_momentum,outward_surface_normal) 
as "angle" parameter to TMM calculations, the dot product is -cos(aoi)

1. -1 at normal incidence against surface_normal, inwards going, aoi = 0 
2. +1 at normal incidence with the surface_normal, outwards going  
3.  0 at glancing incidence (90 deg AOI) : potential for math blowouts here 
4. sign of dot product indicates when must flip the stack of parameters
5. angle scan plots can then use aoi 0->180 deg, which is -cos(aoi) -1->1   
   (will there be continuity across the turnaround ?)

"""

import os, numpy as np, matplotlib as mp
SIZE = np.array([1280, 720]) 

if __name__ == '__main__':

    fig, ax = mp.pyplot.subplots(figsize=SIZE/100.)

    #aoi = np.linspace(0, np.pi/2, 91 )
    aoi = np.linspace(0,  np.pi, 181 )

    aoi_deg = aoi*180/np.pi 

    ax.plot( aoi_deg, -np.cos(aoi), label="-cos(aoi) vs aoi_deg" ) 
    ax.plot( aoi_deg, np.cos(aoi), label="cos(aoi) vs aoi_deg" ) 
    ax.plot( aoi_deg, np.sin(aoi), label="sin(aoi) vs aoi_deg" ) 

    # upper/center/lower right/left 
    ax.legend(loc=os.environ.get("LOC", "lower center"))
    fig.show()


