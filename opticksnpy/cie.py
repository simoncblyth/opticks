#!/usr/bin/env python
"""
Conversion of the binned wavelength spectra into XYZ (using 
CIE weighting functions) and then RGB produces a spectrum

[FIXED] Unphysical color repetition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
Uniform scaling by maximal single X,Y,Z or R,G,B 
prior to clipping gets rid of the unphysical color repetition
but theres kinda a between the green and the blue, where cyan 
should be 

    #hRGB_raw /= hRGB_raw[0,:,0].max()  # scaling by maximal red, results in muted spectrum
    #hRGB_raw /= hRGB_raw[0,:,1].max()  # scaling by maximal green,  OK  
    #hRGB_raw /= hRGB_raw[0,:,2].max()  # scaling by maximal blue, similar to green by pumps the blues and nice yellow  

The entire spectral locus is outside sRGB gamut (the triangle),
so all bins are being clipped.

Not clipping produces a psychedelic mess.


[ISSUE] Blue/Green transition looks unphysical 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need better way to handle out of gamut ?

Raw numbers show that green ramps up thru 430..480 nm but
its all negative, so that info is clipped.

::

    In [68]: np.set_printoptions(linewidth=150)

    In [75]: np.hstack([wd[:-1,None],c.raw[0],c.xyz[0],c.rgb[0]])
    Out[75]: 
    array([[  350.   ,     0.   ,     0.016,     0.102,     0.   ,     0.   ,     0.   ,    -0.   ,     0.   ,     0.   ],
           [  370.   ,     0.015,     0.105,     1.922,     0.   ,     0.   ,     0.001,    -0.001,     0.   ,     0.001],
           [  390.   ,     1.873,     0.582,    20.444,     0.001,     0.   ,     0.011,    -0.003,     0.   ,     0.012],
           [  410.   ,    49.306,     2.691,   205.061,     0.028,     0.002,     0.115,     0.03 ,    -0.019,     0.123],
           [  430.   ,   273.393,    10.384,  1386.823,     0.153,     0.006,     0.779,     0.1  ,    -0.105,     0.83 ],
           [  450.   ,   343.75 ,    33.415,  1781.385,     0.193,     0.019,     1.   ,     0.098,    -0.11 ,     1.064],
           [  470.   ,   191.832,    89.944,  1294.473,     0.108,     0.05 ,     0.727,    -0.091,     0.021,     0.764],
           [  490.   ,    32.012,   213.069,   465.525,     0.018,     0.12 ,     0.261,    -0.256,     0.218,     0.253],
           [  510.   ,    16.48 ,   500.611,   155.962,     0.009,     0.281,     0.088,    -0.446,     0.522,     0.036],
           [  530.   ,   159.607,   869.052,    43.036,     0.09 ,     0.488,     0.024,    -0.472,     0.829,    -0.069],
           [  550.   ,   433.715,   994.463,     8.758,     0.243,     0.558,     0.005,    -0.072,     0.812,    -0.095],
           [  570.   ,   772.904,   950.107,     1.308,     0.434,     0.533,     0.001,     0.586,     0.58 ,    -0.084],
           [  590.   ,  1021.039,   762.587,     0.143,     0.573,     0.428,     0.   ,     1.199,     0.248,    -0.055],
           [  610.   ,  1000.205,   500.338,     0.012,     0.561,     0.281,     0.   ,     1.388,    -0.017,    -0.026],
           [  630.   ,   656.21 ,   263.667,     0.001,     0.368,     0.148,     0.   ,     0.966,    -0.079,    -0.01 ],
           [  650.   ,   283.632,   110.045,     0.   ,     0.159,     0.062,     0.   ,     0.421,    -0.038,    -0.004],
           [  670.   ,    80.766,    36.117,     0.   ,     0.045,     0.02 ,     0.   ,     0.116,    -0.006,    -0.002],
           [  690.   ,    17.024,    11.172,     0.   ,     0.01 ,     0.006,     0.   ,     0.021,     0.003,    -0.001]])



Chromatic Adaption
~~~~~~~~~~~~~~~~~~~~

* http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

     

Refs
~~~~~

https://github.com/colour-science/colour/issues/191
http://www.scipy-lectures.org/advanced/image_processing/index.html
http://www.scipy-lectures.org/packages/scikit-image/index.html#scikit-image

http://www.scipy.org/scikits.html
    separate from scipy, but under the "brand"

"""

import os, logging, numpy as np
log = logging.getLogger(__name__)
np.set_printoptions(linewidth=150)

import matplotlib.pyplot as plt
import ciexyz.ciexyz as _cie
from env.graphics.ciexyz.XYZ import Spectrum
from env.graphics.ciexyz.RGB import RGB


class CIE(object):
    def __init__(self, colorspace="sRGB/D65", whitepoint=None):
        cs = RGB(colorspace)
        self.x2r = cs.x2r
        self.whitepoint = whitepoint

    def hist0d_XYZ(self,w, nb=100):

        X = np.sum(_cie.X(w))
        Y = np.sum(_cie.Y(w))
        Z = np.sum(_cie.Z(w))

        hX = np.repeat(X, nb)
        hY = np.repeat(Y, nb)
        hZ = np.repeat(Z, nb)

        raw = np.dstack([hX,hY,hZ])
        self.raw = np.copy(raw)
        return raw

    def hist1d_XYZ(self,w,x,xb):
        hX, hXx = np.histogram(x,bins=xb, weights=_cie.X(w))   
        hY, hYx = np.histogram(x,bins=xb, weights=_cie.Y(w))   
        hZ, hZx = np.histogram(x,bins=xb, weights=_cie.Z(w))   
        assert np.all(hXx == xb) & np.all(hYx == xb ) & np.all(hZx == xb)
        raw = np.dstack([hX,hY,hZ])
        self.raw = np.copy(raw)
        return raw

    def hist2d_XYZ(self,w,x,y,xb,yb):
        bins = [xb,yb]
        hX, hXx, hXy = np.histogram2d(x,y,bins=bins, weights=_cie.X(w))   
        hY, hYx, hYy = np.histogram2d(x,y,bins=bins, weights=_cie.Y(w))   
        hZ, hZx, hZy = np.histogram2d(x,y,bins=bins, weights=_cie.Z(w))   
        assert np.all(hXx == xb) & np.all(hYx == xb ) & np.all(hZx == xb)
        assert np.all(hXy == yb) & np.all(hYy == yb ) & np.all(hZy == yb)
        return np.dstack([hX,hY,hZ])

    def norm_XYZ(self, hXYZ, norm=2, scale=1):
        """
        Trying to find an appropriate way to normalize XYZ values

        0,1,2
              scale by maximal of X,Y,Z 
        3
              scale by maximal X+Y+Z
        4
              scale by Yint of an externally determined whitepoint
              (problem is that is liable to be with very much more light 
              than are looking at...)
        5
              scale by Yint of the spectrum provided, this 
              is also yielding very small X,Y,Z 

        >50
              scale is used, for normalization with Y value
              obtained from the histogram norm identified bin 


        Hmm, some adhoc exposure factor seems unavoidable given the 
        range of intensities so perhaps the adhoc techniques are appropriate after all.

        Initial thinking was that the out-of-gamut problem was tied up with the 
        exposure problem, but they are kinda orthogonal: think vectors in XYZ space,
        the length of the vector doesnt change the hue.  
        """  
        if norm in [0,1,2]:
            nscale = hXYZ[:,:,norm].max()         
        elif norm == 3:
            nscale = np.sum(hXYZ, axis=2).max()   
        elif norm == 4:
            assert not self.whitepoint is None
            nscale = self.whitepoint[4]
        elif norm == 5 or norm > 50:
            nscale = scale
        else:
            nscale = 1 
        pass

        hXYZ /= nscale             
        self.scale = nscale
        self.xyz = np.copy(hXYZ)
        return hXYZ

    def XYZ_to_RGB(self, hXYZ):
        return np.dot( hXYZ, self.x2r.T )

    def hist0d(self, w, norm=2, nb=100):
        hXYZ_raw = self.hist0d_XYZ(w, nb=nb)
        hXYZ = self.norm_XYZ(hXYZ_raw, norm=norm) 
        hRGB =  self.XYZ_to_RGB(hXYZ)
        self.rgb = np.copy(hRGB)
        return hRGB,hXYZ,None

    def hist1d(self, w, x, xb, norm=2):
        hXYZ_raw = self.hist1d_XYZ(w,x,xb)

        if norm == 5:
            scale = np.sum(_cie.Y(w))
        elif norm > 50:
            # assume norm is pointing to a bin, the Y value of which is used for scaling 
            scale = hXYZ_raw[0,norm,1]
        else:
            scale = 1 
        pass

        hXYZ = self.norm_XYZ(hXYZ_raw, norm=norm, scale=scale) 
        hRGB =  self.XYZ_to_RGB(hXYZ)
        self.rgb = np.copy(hRGB)
        return hRGB,hXYZ,xb

    def hist2d(self, w, x, y, xb, yb, norm=2):
        hXYZ_raw = self.hist2d_XYZ(w,x,y,xb,yb)
        self.raw = hXYZ_raw
        hXYZ = self.norm_XYZ(hXYZ_raw, norm=norm) 
        hRGB =  self.XYZ_to_RGB(hXYZ)
        extent = [xb[0], xb[-1], yb[0], yb[-1]]
        return hRGB,hXYZ,extent

    def spectral_plot(self, ax, wd, norm=2):
 
        ndupe = 1000
        w = np.tile(wd, ndupe)
        x = np.tile(wd, ndupe)
        xb = wd 

        hRGB_raw, hXYZ_raw, bx = self.hist1d(w, x, xb, norm=norm)

        hRGB_1d = np.clip(hRGB_raw, 0, 1)
        ntile = 100
        hRGB = np.tile(hRGB_1d, ntile ).reshape(-1,ntile,3)
        extent = [0,2,bx[0],bx[-1]] 

        #interpolation = 'none'
        #interpolation = 'mitchell'
        #interpolation = 'hanning'
        interpolation = 'gaussian'

        ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1, aspect='auto', interpolation=interpolation)
        ax.yaxis.set_visible(True)
        ax.xaxis.set_visible(False)

    def swatch_plot(self, wd, norm=2):
        ndupe = 1000
        w = np.tile(wd, ndupe)
        hRGB_raw, hXYZ_raw, bx = self.hist0d(w, norm=norm)

        hRGB_1d = np.clip(hRGB_raw, 0, 1)
        ntile = 100
        hRGB = np.tile(hRGB_1d, ntile ).reshape(-1,ntile,3)
        extent = [0,2,0,1]
        ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1, aspect='auto')
        ax.yaxis.set_visible(True)
        ax.xaxis.set_visible(False)



def cie_hist1d(w, x, xb, norm=1, colorspace="sRGB/D65", whitepoint=None):
    c = CIE(colorspace, whitepoint=whitepoint)
    return c.hist1d(w,x,xb,norm)

def cie_hist2d(w, x, y, xb, yb, norm=1, colorspace="sRGB/D65", whitepoint=None):
    c = CIE(colorspace, whitepoint=whitepoint)
    return c.hist2d(w,x,y,xb,yb,norm)



class Whitepoint(object):
    def __init__(self, w):
        """
        For spectra close to original (think perfect diffuse reflector) 
        this is expected to yield the characteristic of the illuminant.

        XYZ values must be normalized as clearly simulating more photons
        will give larger values...

        The Yint is hoped to provide a less adhoc way of doing the
        normalization. 
        """
        assert w is not None

        X = np.sum(_cie.X(w))
        Y = np.sum(_cie.Y(w))
        Z = np.sum(_cie.Z(w))

        Yint = Y

        X /= Yint      # normalize such that Y=1
        Y /= Yint
        Z /= Yint

        x = X/(X+Y+Z)  # Chromaticity coordinates 
        y = Y/(X+Y+Z)

        self.wp = np.array([X,Y,Z,Yint,x,y])

    def __repr__(self):
        return str(self.wp)




def whitepoint(wd):
    bb = _cie.BB6K(wd)
    bb /= bb.max()

    X = np.sum( _cie.X(wd)*bb )
    Y = np.sum( _cie.Y(wd)*bb )
    Z = np.sum( _cie.Z(wd)*bb )

    norm = Y 

    # Normalize Y to 1
    X /= norm
    Y /= norm
    Z /= norm

    return [X,Y,Z], norm



def compare_norms(wdom):
    """
    norm 0:X,1:Y 
             look almost identical  
    
    norm 2:Z, 3:X+Y+Z 
              also look the same
    
    0,1 have better yellow and less of a murky gap between green and blue
    """
    c = CIE()

    nplot = 4

    for i, norm in enumerate([0,1,2,3]):
        ax = fig.add_subplot(1,nplot,i+1)
        c.spectral_plot(ax, wdom, norm)

        #ax = fig.add_subplot(1,nplot,2*i+2)
        #c.swatch_plot(wd, norm)


if __name__ == '__main__':
    pass

    plt.close()
    plt.ion()

    wdom = np.arange(350,720,20)


    fig = plt.figure()

    compare_norms(wdom)

    plt.show()


    wp = whitepoint(wdom)

 









