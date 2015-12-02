#!/usr/bin/env python
"""
Good Reference in Chromaticity etc..

Measuring Colour, R.W.G Hunt 
NTULib QC495 H84, (2nd edition) p62, p204 

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize

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
    (not called xyz as need one of the originals, conventionally Y, to be reversible)


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
        XYZW[:,:,1] = np.ones_like(y)
        XYZW[:,:,2] = z/y 
        XYZW[:,:,3] = np.ones_like(y)

        self.xyzw = xyzw
        self.XYZW = XYZW
        self.extent = extent

    def rgba(self, rgb, clip=True):
        m = rgb.x2r_4.T
        if clip:
            RGBA =  np.clip(np.dot( self.XYZW, m ),0,1)  
        else:
            RGBA =  np.dot( self.XYZW, m )  
        pass
        return RGBA

    def incorrect_gamut_plot(self, rgb, clip=True):
        """
        Attempting to map gamut boundaries by RGB value 
        excursions beyond 0,1 seems to not be valid 
        as it yields very small gamuts ? 
        """
        RGBA = self.rgba(rgb, clip)

        b = np.max(RGBA[:,:,:3], axis=2) > 1.
        d = np.min(RGBA[:,:,:3], axis=2) < 0.
        RGBA[b,3] = 0
        RGBA[d,3] = 0

        plt.imshow(RGBA, origin="lower", extent=self.extent, aspect=1, alpha=1, vmin=0, vmax=1)

    def triangle_gamut_plot(self, rgb, clip=True):
        """
        Instead map the gamut by being within the xy space triangle 
        formed by the R,G,B primaries
        """
        RGBA = self.rgba(rgb, clip)

        t = Tri(rgb.xyz[1:4,:2]) 
        b = t.inside(self.xyzw[:,:,:2]) 

        RGBA[b,3] = 1
        RGBA[~b,3] = 0   # mask the out of triangle

        plt.imshow(RGBA,origin="lower",alpha=1,extent=self.extent, aspect=1, vmin=0, vmax=1)
 

if __name__ == '__main__':

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.axis([0,1,0,1])
    plt.plot([0,1],[1,0])

    cs = ChromaticitySpace()

    rgb = RGB("sRGB/D65")
    print rgb.x2r_4

    #cs.incorrect_gamut_plot(rgb)
    cs.triangle_gamut_plot(rgb)
    rgb.plot()

    XYZ.plot()

 



