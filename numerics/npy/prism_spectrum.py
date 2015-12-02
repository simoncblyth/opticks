#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from env.numerics.npy.ana import Evt, Selection
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.prism import Prism, PrismCheck, PrismExpected

import ciexyz.ciexyz as cie
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

    hX, hXx = np.histogram(x,bins=xb, weights=cie.X(w))   
    hY, hYx = np.histogram(x,bins=xb, weights=cie.Y(w))   
    hZ, hZx = np.histogram(x,bins=xb, weights=cie.Z(w))   

    assert np.all(hXx == xb) & np.all(hYx == xb ) & np.all(hZx == xb)

    hXYZ = np.dstack([hX,hY,hZ])

    hXYZ /= hXYZ[0,:,1].max()   # try scale by maximal Y 

    cs = RGB(colorspace)

    hRGB =  np.dot( hXYZ, cs.x2r.T )

    return hRGB,hXYZ,xb



def cie_hist2d(w, x, y, nx=100, ny=100, colorspace="sRGB/D65", clip=True):

    xmin = x.min()
    xmax = x.max()
    if xmin == xmax:
        xmin = -1
        xmax = 1

    ymin = y.min()
    ymax = y.max()
    if ymin == ymax:
        ymin = -1
        ymax =  1

    xb = np.linspace(xmin, xmax, nx)
    yb = np.linspace(ymin, ymax, ny)
    bins = [xb,yb]

    hX, hXx, hXy = np.histogram2d(x,y,bins=bins, weights=cie.X(w))   
    hY, hYx, hYy = np.histogram2d(x,y,bins=bins, weights=cie.Y(w))   
    hZ, hZx, hZy = np.histogram2d(x,y,bins=bins, weights=cie.Z(w))   

    assert np.all(hXx == xb) & np.all(hYx == xb ) & np.all(hZx == xb)
    assert np.all(hXy == yb) & np.all(hYy == yb ) & np.all(hZy == yb)

    hXYZ = np.dstack([hX,hY,hZ])

    cs = RGB(colorspace)
    hRGB =  np.dot( hXYZ, cs.x2r.T )
    if clip:
        hRGB = np.clip(hRGB, 0, 1) 

    extent = [xb[0], xb[-1], yb[0], yb[-1]]
    return hRGB,extent




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    plt.ion()

    evt = Evt(tag="1", det="prism")

    sel = Selection(evt,"BT BT SA")  

    boundary = Boundary("Vacuum///GlassSchottF2")

    prism = Prism("60.,300,300,0", boundary)

    n = boundary.imat.refractive_index(sel.wl)  

    xprism = PrismExpected(prism.a, n)

    pc = PrismCheck(prism, xprism, sel )

    w = pc.wx
    x = pc.p3[:,0]
    y = pc.p3[:,1]
    z = pc.p3[:,2]
    assert np.all(x == 700.)

    fig = plt.figure()

    #bands = [300,800]
    bands = [w.min(),w.max()]

    nc = len(bands) - 1
    nr = 1 
    ntile = 50

    #hRGB,extent = cie_hist2d(w,z,y,nx=1,ny=100, colorspace="sRGB/D65", clip=True)

    for ib in range(len(bands)-1):

        ax= fig.add_subplot(nr,nc,ib+1)

        b = np.logical_and( w > bands[ib], w < bands[ib+1] )

        nx = 100 

        hRGB_raw,hXYZ_raw, bx = cie_hist1d(w[b],y[b], nx=nx, colorspace="sRGB/D65")

        # uniform scaling by maximal single colors, prior to clipping 
        # gets rid of the unphysical color repetition
        # but theres kinda a gap between the green and the blue 

        #hRGB_raw /= hRGB_raw[0,:,0].max()  # scaling by maximal red, results in muted spectrum
        #hRGB_raw /= hRGB_raw[0,:,1].max()  # scaling by maximal green,  OK  
        #hRGB_raw /= hRGB_raw[0,:,2].max()  # scaling by maximal blue, similar to green by pumps the blues and nice yellow  
    

        hRGB_1 = np.clip(hRGB_raw, 0, 1)

        hRGB_2 = np.tile(hRGB_1, ntile ).reshape(-1,ntile,3)

        extent = [0,ntile,bx[0],bx[-1]] 

        ax.imshow(hRGB_2, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1)

        #ax.scatter(y[b][::10],w[b][::10],c=hRGB_1[0][::10])




