#!/usr/bin/env python
"""
Good Reference in Chromaticity etc..

Measuring Colour, R.W.G Hunt 
NTULib QC495 H84, (2nd edition) p62, p204 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from env.graphics.ciexyz.XYZ import XYZ
from env.graphics.ciexyz.RGB import RGB 
from env.graphics.ciexyz.tri import Tri 


def hsv_plot():
    """
    # http://stackoverflow.com/questions/10787103/2d-hsv-color-space-in-matplotlib
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



class ChromaticitySpace(object):
    """
    http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html

    ::

              x Y
        X =   ---
                y

        Y = Y

              (1 - x - y)   Y       z Y
        Z =  ----------------   =   ----
                    y                y 



    Chromaticity coordinates, xyY color space

                     X
        x =    ---------------       
                 X + Y + Z

                     Y
        y =    ---------------       
                 X + Y + Z



                     Z
        z =    ---------------    = 1 - x - y     
                 X + Y + Z


    """
    def __init__(self, nbin=500):

        start = 1./nbin  # skip zero to avoid infinity, nan handling 
        end = 1.
        step = complex(0,nbin)


        extent = [start, end, start, end]
        y, x = np.mgrid[start:end:step,start:end:step]
        z = np.ones_like(y) - x - y 
        w = np.ones_like(y)
        xyzw = np.dstack((x,y,z,w))

        # the above grid is xyY, need to convert into XYZ before
        # can apply the color space transforms ?
        # BUT: what normalization to use ?   Using Y=1

        XYZW = np.empty_like(xyzw)
        XYZW[:,:,0] = x/y 
        XYZW[:,:,1] = np.ones_like(XYZW[:,:,1])
        XYZW[:,:,2] = z/y 
        XYZW[:,:,3] = np.ones_like(XYZW[:,:,3])


        # maybe should not divide by this sum, as that returns to x,y,z ? 
        XYZsum = np.repeat(np.sum(XYZW[:,:,:3], axis=2),3).reshape(nbin,nbin,3)
        NXYZW = np.empty_like(xyzw)
        NXYZW[:,:,:3] = XYZW[:,:,:3]/XYZsum
        NXYZW[:,:,3] = np.ones_like(NXYZW[:,:,3])

        self.xyzw = xyzw
        self.XYZW = XYZW
        self.XYZsum = XYZsum
        self.NXYZW = NXYZW
        self.extent = extent


    def incorrect_gamut_plot(self, rgb):
        """
        Attempting to map gamut boundaries by RGB value 
        excursions beyond 0,1 seems to not be valid 
        yields very small gamuts ? 
        """

        RGBA = np.dot(self.NXYZW, rgb.x2r_4  )

        b = np.max(RGBA[:,:,:3], axis=2) > 1.
        d = np.min(RGBA[:,:,:3], axis=2) < 0.
        RGBA[b,3] = 0.1
        RGBA[d,3] = 0.1

        plt.imshow(RGBA, origin="lower", extent=[0,0.8,0,0.9], aspect=1, alpha=0.5, vmin=0, vmax=1)
 

    def triangle_gamut_plot(self, rgb):
        pass



if __name__ == '__main__':

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.axis([0,1,0,1])
    plt.plot([0,1],[1,0])

    cs = ChromaticitySpace()

    for space in RGB.spaces()[0:1]:
        rgb = RGB(space)
        cs.triangle_gamut_plot(rgb)

        rgb.plot()

        t = Tri(rgb.xyz[1:4,:2]) 

        b = t.inside(cs.xyzw[:,:,:2])

        RGBA =  np.dot( cs.XYZW, rgb.x2r_4 )

        #RGBA = np.zeros_like(cs.XYZW)
        #RGBA[:,:,0] = 1
        #RGBA[:,:,1] = 1
        #RGBA[:,:,2] = 1

        RGBA[b,3] = 1
        RGBA[~b,3] = 0

        plt.imshow(RGBA,origin="lower",alpha=1,extent=cs.extent, aspect=1)
    pass


    xyz = XYZ()



 



