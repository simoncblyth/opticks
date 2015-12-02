#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

import ciexyz.ciexyz as _cie
from env.graphics.ciexyz.XYZ import Spectrum
from env.graphics.ciexyz.RGB import RGB


def cie_hist1d(w, x, nx=100, colorspace="sRGB/D65"):
    """
    :param w: array of wavelengths
    :param x: coordinate
    :param nx: number of bins

    :return hRGB: 
    :return hb: 


    Scatter plot of wavelength against the appropriate prism coordinate
    shows expected behaviour, except at the endpoints with outlier at high wavelength 
    and a cliff drop off at low wavelenth. 
    Perhaps these are due to restricted range of refractive index values or 
    problem with spectral edges of the black body light source. 

    TODO: check undispersed spot, extend refractive index values 

    Conversion of the binned wavelength spectra into XYZ (using 
    CIE weighting functions) and then RGB produces a spectrum, BUT
    are seeing unexpected repetition of colors.  

    Presumably the repetition arises from the trivial 0,1 
    clipping of RGB values.
    The entire spectral locus is outside sRGB gamut (the triangle),
    so all bins are being clipped.

    Not clipping produces a psychedelic mess.
   
    Better way to handle out of gamut ?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     
    https://github.com/colour-science/colour/issues/191
    http://www.scipy-lectures.org/advanced/image_processing/index.html
    http://www.scipy-lectures.org/packages/scikit-image/index.html#scikit-image

    http://www.scipy.org/scikits.html
        separate from scipy, but under the "brand"

    """
    xmin = x.min()
    xmax = x.max()
    if xmin == xmax:
        xmin = -1
        xmax = 1

    xb = np.linspace(xmin, xmax, nx)

    hX, hXx = np.histogram(x,bins=xb, weights=_cie.X(w))   
    hY, hYx = np.histogram(x,bins=xb, weights=_cie.Y(w))   
    hZ, hZx = np.histogram(x,bins=xb, weights=_cie.Z(w))   

    assert np.all(hXx == xb) & np.all(hYx == xb ) & np.all(hZx == xb)

    hXYZ = np.dstack([hX,hY,hZ])

    hXYZ /= hXYZ[0,:,1].max()   # try scale by maximal Y 

    cs = RGB(colorspace)

    hRGB =  np.dot( hXYZ, cs.x2r.T )

    return hRGB,hXYZ,xb



def cie_hist2d(w, x, y, nx=100, ny=100, colorspace="sRGB/D65"):

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    xb = np.linspace(xmin, xmax, nx)
    yb = np.linspace(ymin, ymax, ny)
    bins = [xb,yb]

    hX, hXx, hXy = np.histogram2d(x,y,bins=bins, weights=_cie.X(w))   
    hY, hYx, hYy = np.histogram2d(x,y,bins=bins, weights=_cie.Y(w))   
    hZ, hZx, hZy = np.histogram2d(x,y,bins=bins, weights=_cie.Z(w))   

    assert np.all(hXx == xb) & np.all(hYx == xb ) & np.all(hZx == xb)
    assert np.all(hXy == yb) & np.all(hYy == yb ) & np.all(hZy == yb)

    hXYZ = np.dstack([hX,hY,hZ])

    hXYZ /= hXYZ[:,:,1].max()   # try scale by maximal Y 

    cs = RGB(colorspace)
    hRGB =  np.dot( hXYZ, cs.x2r.T )

    extent = [xb[0], xb[-1], yb[0], yb[-1]]
    return hRGB,hXYZ,extent



if __name__ == '__main__':
    pass
