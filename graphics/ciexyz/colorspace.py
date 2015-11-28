#!/usr/bin/env python
"""
Good Reference in Chromaticity etc..

NTULib QC495 H84, Measuring Colour second edition, R.W.G Hunt 

p62
p204 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from env.graphics.ciexyz.XYZ import XYZ, xyz2rgb 

def hsv_plot():
    """
    # http://stackoverflow.com/questions/10787103/2d-hsv-color-space-in-matplotlib

    Shapes of the arrays

    V   (100, 300)
    H   (100, 300)
    S   (100, 300)

    HSV (100, 300, 3)
    RGB (100, 300, 3)

    """
    V, H = np.mgrid[0:1:100j, 0:1:300j]   # complex step means inclusive
    S = np.ones_like(V)

    HSV = np.dstack((H,S,V))   # depth stacking
    RGB = hsv_to_rgb(HSV)

    plt.imshow(RGB, origin="lower", extent=[0, 360, 0, 1], aspect=150)
    plt.xlabel("H")
    plt.ylabel("V")
    plt.title("$S_{HSV}=1$")
    plt.show()



if __name__ == '__main__':

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(121)

    xyz = XYZ()

    nbin = 500j
    Y, X = np.mgrid[0:1:nbin, 0:1:nbin]
    Z = np.ones_like(X) - X - Y 
    W = np.ones_like(X)
    cs = np.dstack((X,Y,Z,W))

    spaces = []
    #spaces.append("sRGB/D65")
    #spaces.append("WideGamutRGB/D50")
    #spaces.append("CIE_RGB/E")
    spaces.append("AdobeRGB/D65")

    for space in spaces:
        x2r,r2x = xyz2rgb(space)
        x2r_w = np.identity(4)
        x2r_w[:3,:3] = x2r

        RGBA = np.dot(cs, x2r_w  )

        # instead of clipping out of gamut, mask them
        b = np.max(RGBA[:,:,:3], axis=2) > 1.
        d = np.min(RGBA[:,:,:3], axis=2) < 0.
        RGBA[b,3] = 0
        RGBA[d,3] = 0

        plt.imshow(RGBA, origin="lower", extent=[0,0.8,0,0.9], aspect=1, alpha=0.5)
    pass



    sl = slice(150,-150) 
    #sl = slice(0,len(xyz.w))
    plt.plot( xyz.nmono[sl,0], xyz.nmono[sl,1] )
 
    ll = slice(150,-150,10) 
    wls = xyz.w[ll]
    xyn = xyz.nmono[ll,:2]
 
    plt.scatter( xyn[:,0], xyn[:,1] )

    for i in range(len(wls)):
        plt.annotate(wls[i], xy = xyn[i], xytext = (0.5, 0.5), textcoords = 'offset points')




