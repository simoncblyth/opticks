#!/usr/bin/env python
"""
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from opticks.ana.base import opticks_environment
from opticks.ana.evt import Evt, costheta_, cross_
from opticks.ana.proplib import Boundary   
from opticks.ana.fresnel import fresnel_factor
from opticks.ana.cie  import CIE
from opticks.ana.xrainbow import XRainbow
from opticks.ana.rainbow  import Rainbow, Rainbows

deg = np.pi/180.



def deviation_plot_0(evt):
    """
    looking at deviation angle without assuming a particular rainbow, 
    to allow seeing all bows at once and comparing intensities
    """

    dv0 = evt.deviation_angle()
    w = evt.wavelength
    b = w > 0 

    d = dv0/deg
    db = np.arange(0,360,1)

    cnt, bns, ptc = plt.hist(d, bins=db, log=True)

    cie = CIE(colorspace="sRGB/D65", whitepoint=evt.whitepoint)
    

    hRGB_raw, hXYZ_raw, bx= cie.hist1d(w[b],d[b], db, norm=1)
    hRGB = np.clip(hRGB_raw, 0, 1)

    hRGB[0] /= np.repeat(np.max( hRGB[0], axis=1), 3).reshape(-1,3)  

    # pumping exposure like this, with a different factor for every bin creates a mess 
    # brains expect single exposures, need to pick a reference bin (eg 1st bow)
    # and pump to expose that 

    for i in range(len(ptc)):
        if cnt[i] > 1000:
            ptc[i].set_facecolor(hRGB[0,i])
            ptc[i].set_edgecolor(hRGB[0,i])



def bow_angle_plot(bows):
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    nk = max(map(int,bows.keys()))
    for k in range(1,nk+1):
        dvr = bows[k].xbow.dvr/deg
        rect = Rectangle( (dvr[0], ymin), dvr[1]-dvr[0], ymax-ymin, alpha=0.1 ) 
        ax.add_patch(rect)
        ax.annotate( "%s" % k, xy=((dvr[0]+dvr[1])/2, 2), color='red')
    pass


def deviation_plot(evt, bows):

    dv0 = evt.deviation_angle()
    w = evt.wavelength
    b = w > 0 

    d = dv0/deg
    db = np.arange(0,360,1)

    ax = fig.add_subplot(3,1,1)
    ax.set_xlim(0,360)
    ax.set_ylim(1,1e5)

    cnt, bns, ptc = ax.hist(d, bins=db,  log=True, histtype='step')
    ymin, ymax = ax.get_ylim()
    dy = ymax - ymin

    bow_angle_plot(bows) 


    ax.annotate("Rainbow visible ranges",xy=(250,2), color='red') 


    cie = CIE(colorspace="sRGB/D65", whitepoint=evt.whitepoint)

    # expose for bow1 bin 138 
    hRGB_raw_H, hXYZ_raw_H, bx_H = cie.hist1d(w[b],d[b], db, norm=138)
    hRGB_H = np.tile(np.clip(hRGB_raw_H,0,1), 50).reshape(-1,50,3)

    # expose for bow2 bin 232
    hRGB_raw_L, hXYZ_raw_L, bx_L= cie.hist1d(w[b],d[b], db, norm=232)
    hRGB_L = np.tile(np.clip(hRGB_raw_L,0,1), 50).reshape(-1,50,3)

    extent = [db[0],db[-1],0,2] 

    #interpolation = 'none'
    #interpolation = 'mitchell'
    interpolation = 'gaussian'

    ax = fig.add_subplot(3,1,2)
    ax.imshow( np.swapaxes(hRGB_H,0,1), origin="lower", extent=extent, alpha=1, vmin=0, vmax=1, aspect='auto', interpolation=interpolation)
    ax.yaxis.set_visible(False)

    ax = fig.add_subplot(3,1,3)
    ax.imshow( np.swapaxes(hRGB_L,0,1), origin="lower", extent=extent, alpha=1, vmin=0, vmax=1, aspect='auto', interpolation=interpolation)
    ax.yaxis.set_visible(False)






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    plt.ion()
    plt.close()

    #intensity_plot() 
    #scattering_angle_plot() 

if 1:
    boundary = Boundary("Vacuum///MainH2OHale")

    # created with ggv-;ggv-rainbow green    etc..
    # huh the blue is coming out magenta 
    # (maximal Z at 
    white, red, green, blue, spol, ppol = "1","2","3","4","5", "6"

    #w_evt = dict(tag=white, det="rainbow", label="3M")
    p_evt = dict(tag=ppol, det="rainbow", label="P")
    s_evt = dict(tag=spol, det="rainbow", label="S")

    #n = boundary.imat.refractive_index(w_evt.wavelength) 
    #navg = (n.min() + n.max())/2.

    nred = 1.331
    xfa = XFrac(nred)

    #w_bows = Rainbows(w_evt, boundary, nk=6)
    p_bows = Rainbows(p_evt, boundary, nk=6)
    s_bows = Rainbows(s_evt, boundary, nk=6)

    #w_bows.selection_counts()
    p_bows.selection_counts()
    s_bows.selection_counts()


    bow = s_bows[1]

    #w = bow.w 
    dv = bow.dv
    x = bow.pos[:,0]
    y = bow.pos[:,1]
    z = bow.pos[:,2]
    r = np.sqrt(y*y + z*z)

    xbow = bow.xbow 
    n = xbow.n
    xv = xbow.dv





if 0:
    fig = plt.figure()
    fig.suptitle("Simulated Deviation Angles of 3M Optical Photons Incident on Spherical Water Droplet")
    deviation_plot(p_evt, bows)


if 0:
    fig = plt.figure()
    fig.suptitle("Interpolated Spectrum Images of 1st 6 Rainbows (3M Simulated Photons incident on water droplet)")
    for i,k in enumerate(range(1,nk+1)):
        ax = fig.add_subplot(1,nk,i+1)
        bows[k].cieplot_1d(norm=1)

if 0:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bows[1].cieplot_1d(norm=1)


if 0:
    fig = plt.figure()
    for k in range(1,nk+1):
        ax = fig.add_subplot(1,nk,k)
        bows[k].hist_1d()

if 0:
    fig = plt.figure()
    for i,k in enumerate(range(1,nk+1)):
        ax = fig.add_subplot(2,3,i+1)
        bows[k].plot_1d()


if 0:
    fig = plt.figure()

    ax = fig.add_subplot(131)
    bow.cieplot_2d()

    ax = fig.add_subplot(132)
    bow.hist_2d()

    #ax = fig.add_subplot(133)
    #bow.plot_2d()





