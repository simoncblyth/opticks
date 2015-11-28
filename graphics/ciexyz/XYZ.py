#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import ciexyz.ciexyz as c   # see "ufunc-cd ciexyz"

def XYZToRelativeXYZ(XYZ):
    X,Y,Z = XYZ
    print "X,Y,Z %s %s %s " % (X,Y,Z)  
    x =  X/(X+Y+Z)   # relative color : chromaticity coordinates
    y =  Y/(X+Y+Z)
    z =  Z/(X+Y+Z)
    print "x,y,z %s %s %s " % (x,y,z)  
    return x,y,z

def RelativeXYZToRGB(xyz):
    """
    mdfind XYZToRGB -> /Developer/NVIDIA/CUDA-7.0/doc/pdf/NPP_Library.pdf
    Search PDF for XYZToRGB yields:below XYZ to RGB color conversion method

    """
    x,y,z = xyz

    x = x/255.
    y = y/255.
    z = z/255.

    R = 255.*min(1.0, 3.240479 * x - 1.53715  * y - 0.498535 * z)
    G = 255.*min(1.0,-0.969256 * x + 1.875991 * y + 0.041556 * z)
    B = 255.*min(1.0, 0.055648 * x - 0.204043 * y + 1.057311 * z)

    return R,G,B


def xyz2rgb(name="sRGB/D65"):
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

    space = {} 
    space["sRGB/D65"] = """
 3.2404542 -1.5371385 -0.4985314
-0.9692660  1.8760108  0.0415560
 0.0556434 -0.2040259  1.0572252
"""
    space["WideGamutRGB/D50"] = """
 1.4628067 -0.1840623 -0.2743606
-0.5217933  1.4472381  0.0677227
 0.0349342 -0.0968930  1.2884099
"""
    space["AdobeRGB/D65"] = """
2.0413690 -0.5649464 -0.3446944
-0.9692660  1.8760108  0.0415560
 0.0134474 -0.1183897  1.0154096
"""
    space["AppleRGB/D65"] = """
 2.9515373 -1.2894116 -0.4738445
-1.0851093  1.9908566  0.0372026
 0.0854934 -0.2694964  1.0912975
"""
    space["BruceRGB/D65"] = """
2.7454669 -1.1358136 -0.4350269
-0.9692660  1.8760108  0.0415560
 0.0112723 -0.1139754  1.0132541
"""
    space["CIE_RGB/E"] = """
2.3706743 -0.9000405 -0.4706338
-0.5138850  1.4253036  0.0885814
 0.0052982 -0.0146949  1.0093968
"""

   
    xyz2rgb_ = np.fromstring(space[name],sep=" ").reshape(3,3)
    rgb2xyz_ = np.linalg.inv(xyz2rgb_)

    return xyz2rgb_, rgb2xyz_


class XYZ(object):
    """
    http://www.fourmilab.ch/documents/specrend/
    """
    def __init__(self, w=None):
        if w is None:
            w = np.linspace(300,800,501) 
        pass

        xf = c.X(w) # CIE color matching functions
        yf = c.Y(w)
        zf = c.Z(w)
        yint = np.sum(yf)
 
        tf = xf+yf+zf

        xr = xf/tf
        yr = yf/tf
        zr = zf/tf

        self.w = w

        mxyz = np.empty((len(w),3))
        mxyz[:,0] = xf
        mxyz[:,1] = yf
        mxyz[:,2] = zf
        mxyz /= yint
        self.mono = mxyz  # color matching functions normalized by y integral 


        # (x,y,z)/(x+y+z)
        # repeat sum thrice, and temporarily flatten to apply 
        nmono = (mxyz.reshape(-1)/np.repeat(np.sum(mxyz, axis=1),3)).reshape(-1,3)
        self.nmono = nmono

        self.xf = xf
        self.yf = yf
        self.zf = zf
        self.yint = yint

        self.xr = xr
        self.yr = yr
        self.zr = zr

        self.bb5k = c.BB5K(w)   # blackbody spectrum
        self.bb6k = c.BB6K(w)

    def mono_spectrum(self, wl):
        imon = np.where(self.w==wl)[0][0] 
        smon = np.zeros_like(self.w)
        smon[imon] = 1 
        return smon

    def spectrumToXYZ(self,s):
        assert len(self.w) == len(s)
        X = np.sum(s*self.xf) 
        Y = np.sum(s*self.yf) 
        Z = np.sum(s*self.zf) 
        return np.array([X,Y,Z])/self.yint

    def spectrumToRGB(self,s):
        XYZ = self.spectrumToXYZ(s)
        xyz = XYZToRelativeXYZ(XYZ)
        RGB = RelativeXYZToRGB(xyz)
        print "RGB %s " % repr(RGB) 
        return RGB



def xyz_plot(xyz,sl=slice(150,-150)):
    plt.plot( xyz.nmono[sl,0], xyz.nmono[sl,1] )

def xyz_mono_check(xyz, clip=True):
    x2r_0,r2x_0 = xyz2rgb("WideGamutRGB/D50")
    x2r_1,r2x_1 = xyz2rgb("sRGB/D65")

    for wl in np.arange(450,630,10,dtype=np.float64):

        i = np.where(xyz.w == wl)[0][0]

        p = xyz.mono_spectrum(wl)
        o = xyz.spectrumToXYZ(p)
        c = o/o.sum()

        m = xyz.mono[i]
        m /= m.sum()

        n = xyz.nmono[i]

        assert np.allclose(n,c)
        assert np.allclose(n,m)

        rgb_0 = np.dot(n,x2r_0) 
        rgb_1 = np.dot(n,x2r_1) 

        if clip:
            rgb_0 = np.clip(rgb_0,0,1)
            rgb_1 = np.clip(rgb_1,0,1)


        print "wl %10.1f o %s c %s m %s n %s rgb0 %s rgb1 %s  " % (wl, str(o), str(c), str(m), str(n), str(rgb_0), str(rgb_1))



def xyz_bb_check(xyz):

    for name in ["bb5k","bb6k"]:
        s = getattr(xyz,name)
        o = xyz.spectrumToXYZ(s)
        c = o/o.sum()
        x2r,r2x = xyz2rgb()
        rgb = np.dot(c,x2r) 
        print "%s XYZ %s xyz %s rgb %s " % (name, str(o), str(c), str(rgb))
    pass


def xyz2rgb_check(name):

    print name 

    x2r,r2x = xyz2rgb(name)
 
    W = np.array([1,1,1])
    R = np.array([1,0,0])
    G = np.array([0,1,0])
    B = np.array([0,0,1])
    C = np.array([0,1,1])
    M = np.array([1,0,1])
    Y = np.array([1,1,0])
    K = np.array([0,0,0])

    for r in [W,R,G,B,C,M,Y,K]:
        x = np.dot(r, r2x)
        rr = np.dot( x, x2r )
        print "r %s x %s rr %s " % (str(r), str(x), str(rr))




if __name__ == '__main__':
    pass

    xyz = XYZ()
    xyz_plot(xyz)
    xyz_mono_check(xyz)
    xyz_bb_check(xyz)

    xyz2rgb_check("sRGB/D65")
    xyz2rgb_check("WideGamutRGB/D50")


