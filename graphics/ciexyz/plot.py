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


class XYZ(object):
    """
    http://www.fourmilab.ch/documents/specrend/
    """
    def __init__(self, w):
        self.w = w
        self.xf = c.X(w) # CIE color matching functions
        self.yf = c.Y(w)
        self.zf = c.Z(w)
        self.bb5k = c.BB5K(w)   # blackbody spectrum
        self.bb6k = c.BB6K(w)

    def spectrumToXYZ(self,s):
        assert len(self.w) == len(s)
        X = np.sum(s*self.xf) 
        Y = np.sum(s*self.yf) 
        Z = np.sum(s*self.zf) 
        return X,Y,Z

    def spectrumToRGB(self,s):
        XYZ = self.spectrumToXYZ(s)
        xyz = XYZToRelativeXYZ(XYZ)
        RGB = RelativeXYZToRGB(xyz)
        print "RGB %s " % repr(RGB) 
        return RGB



def plot_XYZ(xyz):
    fig = plt.figure()
    w = xyz.w 
    plt.plot(w,xyz.xf,label="X", c="r")
    plt.plot(w,xyz.yf,label="Y", c="g")
    plt.plot(w,xyz.zf,label="Z", c="b")
    plt.legend()
    fig.show()

def plot_BB(xyz):
    fig = plt.figure()
    w = xyz.w 
    plt.plot(w,xyz.bb5k,label="5K", c="r")
    plt.plot(w,xyz.bb6k,label="6K", c="b")
    plt.legend()
    fig.show()




if __name__ == '__main__':

    w = np.linspace(300,800,501)

    plt.ion()

    xyz = XYZ(w)

    plot_XYZ(xyz)
    plot_BB(xyz)

    flat = np.ones_like(w)  # flat spectrum, will it be white ?  nope dull grey

    RGB = xyz.spectrumToRGB(flat)
    print "flat spectrum RGB:%s " % repr(RGB) 

    bb5k = c.BB5K(w)
    RGB = xyz.spectrumToRGB(bb5k)
    print "bb5k spectrum RGB:%s " % repr(RGB) 

    bb6k = c.BB6K(w)
    RGB = xyz.spectrumToRGB(bb6k)
    print "bb6k spectrum RGB:%s " % repr(RGB) 




