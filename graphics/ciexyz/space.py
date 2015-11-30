#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

import numpy as np

class Space(object):
    """
    pbrt p273
    
    rgb[0] =  3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
    rgb[1] = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
    rgb[2] =  0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];

    http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    Comparing with many matrices listed on above page the pbrt matrix corresponds to sRGB/D65

    http://ninedegreesbelow.com/photography/xyz-rgb.html
    RGB color spaces use 5 XYZ coordinates for R,G,B primaries and W,B white and black points

    http://www.mathworks.com/help/images/ref/xyz2rgb.html

    How to avoid negative RGB ?  Use wider gamut space ? Otherwise how to clip ?
    """

    s = {} 
    s["sRGB/D65"] = """
 3.2404542 -1.5371385 -0.4985314
-0.9692660  1.8760108  0.0415560
 0.0556434 -0.2040259  1.0572252
"""
    s["WideGamutRGB/D50"] = """
 1.4628067 -0.1840623 -0.2743606
-0.5217933  1.4472381  0.0677227
 0.0349342 -0.0968930  1.2884099
"""
    s["AdobeRGB/D65"] = """
2.0413690 -0.5649464 -0.3446944
-0.9692660  1.8760108  0.0415560
 0.0134474 -0.1183897  1.0154096
"""
    s["AppleRGB/D65"] = """
 2.9515373 -1.2894116 -0.4738445
-1.0851093  1.9908566  0.0372026
 0.0854934 -0.2694964  1.0912975
"""
    s["BruceRGB/D65"] = """
2.7454669 -1.1358136 -0.4350269
-0.9692660  1.8760108  0.0415560
 0.0112723 -0.1139754  1.0132541
"""
    s["CIE_RGB/E"] = """
2.3706743 -0.9000405 -0.4706338
-0.5138850  1.4253036  0.0885814
 0.0052982 -0.0146949  1.0093968
"""

    @classmethod
    def space(cls, name): 
        xyz2rgb_ = np.fromstring(cls.s[name],sep=" ").reshape(3,3)
        rgb2xyz_ = np.linalg.inv(xyz2rgb_)
        return xyz2rgb_, rgb2xyz_



if __name__ == '__main__':
    pass

    for key in Space.s.keys():
        x2r, r2x = Space.space(key)
        print "\n\n"
        print "\n[%s] XYZ -> RGB\n" % key, x2r
        print "\n[%s] RGB -> XYZ\n" % key, r2x



