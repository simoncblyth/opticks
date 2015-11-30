#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt


class sRGB(object):
    """
    Getting to grips with color spaces by obtaining the 
    sRGB to XYZ matrices from the chromaticity of R,G,B primaries
    and white point

    The transformations between XYZ and linear RGB are 
    below obtained following technique from 

    http://www.ryanjuckett.com/programming/rgb-color-space-conversion/
    """
    xy = np.array([
             [0.3127, 0.3290],
             [0.64,0.33],
             [0.30,0.60],
             [0.15,0.06]
             ])
 
    names = ["W", "R","G","B"]

    def plot(self):
        xyz = self.xyz
        plt.scatter( xyz[:,0], xyz[:,1] )
        for i in range(len(self.names)):
            plt.annotate(self.names[i], xy = xyz[i][:2], xytext = (0.5, 0.5), textcoords = 'offset points')

    def __init__(self): 
        """
        """
        # from xy to xyz by adding a z=1-x-y column 
        xyz = np.empty((len(self.xy),3))
        xyz[:,:2] = self.xy
        xyz[:,2] = np.ones_like(xyz[:,2]) - np.sum(xyz[:,:2], axis=1)

        # white point xyz coordinate to an XYZ coordinate by using a Y luminance value of 1.
        wpY = xyz[0][1]    # 0.32900000
        wXYZ = xyz[0]/wpY    # scale to make white point Y luminance of 1: [ 0.95 ,  1.   ,  1.089] 

        # Solve for the (X+Y+Z) scalar values that will convert each xyz primary to XYZ space.
        XYZSum = np.dot( np.linalg.inv(xyz[1:4].T) , wXYZ ) 
     
        # Reconstruct the matrix M which transforms from linear sRGB space to XYZ space.
        linear_sRGB_to_XYZ = np.dot( xyz[1:4].T, np.diag(XYZSum) )
        XYZ_to_linear_sRGB = np.linalg.inv(linear_sRGB_to_XYZ)

        self.xyz = xyz 
        self.wXYZ = wXYZ 
        self.XYZSum = XYZSum
        self.x2r = XYZ_to_linear_sRGB
        self.r2x = linear_sRGB_to_XYZ
        self.wpY = wpY

    @classmethod
    def check(cls):
        """
        Sanity check the matrix by conversion of RGB into XYZ 
        """ 
        srgb = cls()

        w = [1,1,1]
        r = [1,0,0]
        g = [0,1,0]
        b = [0,0,1]

        wXYZ = np.dot( srgb.r2x, w )   
        rXYZ = np.dot( srgb.r2x, r )   
        gXYZ = np.dot( srgb.r2x, g )   
        bXYZ = np.dot( srgb.r2x, b ) 

        wxyz = wXYZ/wXYZ.sum()
        rxyz = rXYZ/rXYZ.sum()
        gxyz = gXYZ/gXYZ.sum()
        bxyz = bXYZ/bXYZ.sum()

        log.info("w %15s wXYZ %20s wxyz %20s (expect) %20s " % (w,wXYZ,wxyz,srgb.xyz[0]))
        log.info("r %15s rXYZ %20s rxyz %20s (expect) %20s " % (r,rXYZ,rxyz,srgb.xyz[1]))
        log.info("g %15s gXYZ %20s gxyz %20s (expect) %20s " % (g,gXYZ,gxyz,srgb.xyz[2]))
        log.info("b %15s bXYZ %20s bxyz %20s (expect) %20s " % (b,bXYZ,bxyz,srgb.xyz[3]))

        assert np.allclose(srgb.xyz[0], wxyz) 
        assert np.allclose(srgb.xyz[1], rxyz) 
        assert np.allclose(srgb.xyz[2], gxyz) 
        assert np.allclose(srgb.xyz[3], bxyz) 


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)


    sRGB.check()
    srgb = sRGB()


    from RGB import RGB

    rgb = RGB()
    rgb.table()

    assert (srgb.x2r-rgb.x2r).max() < 0.001
    assert (srgb.x2r-rgb.x2r).min() > -0.001




