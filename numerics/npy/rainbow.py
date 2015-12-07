#!/usr/bin/env python
"""

Kurt Nassau, The Physics and Chemistry of Color

p219, Dispersion with single reflection (in spherical raindrop)
      (shadow of head at center of bow)

first bow
      V:41 R:43 degree cone around the eye-antisolar point axis

2nd bow 
      R:50, V:54 

Alexanders dark band, between the 1st and 2nd bows 
(due to no rays below min deviation for each bow)


"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from env.numerics.npy.ana import Evt, Selection, costheta_, cross_
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.cie  import CIE
deg = np.pi/180.


class XRainbow(object):
    def __init__(self, w, boundary, k=1 ):
        """
        :param w: wavelength array
        :param boundary: instance
        :param k: 1.. rainbow index, -1 direct reflection 

        Attributes:

        i 
            incident angle of minimum deviation 
        d 
            total deviation angle at minimum deviation
        n 
            refractive indices corresponding to wavelength array

        
        Using derivations from: Jearl D. Walker
        "Multiple rainbows from single drops of water and other liquids",  

        * http://www.patarnott.com/atms749/pdf/MultipleRainbowsSingleDrops.pdf


        There is symmetry about the ray at normal incidence so consider
        a half illuminated drop.

        Deviation angles are calculated in range 0:360 degrees 

           k    red    blue
           1   137.63  139.35
           2   230.37  233.48


        0:180 
             signifies rays exiting in hemisphere opposite 
             to the incident hemisphere

        180:360 
             signifies rays exiting in same hemisphere  
             as incidence
 
        """
        n = boundary.imat.refractive_index(w) 


        if k == -1:
            i = 0
            r = 0
            dv = np.ones(len(w))*np.pi
        else:
            # incident, refracted angles at the minimum deviation
            # NB these are arrays corresponding to all refractive indices of the 
            # wavelengths of the sample
            i = np.arccos( np.sqrt((n*n - 1.)/(k*(k+2.)) ))
            r = np.arcsin( np.sin(i)/n )
            dv = k*np.pi + 2*i - 2*r*(k+1)
        pass

        self.w = w 
        self.n = n
        self.k = k
        self.i = i
        self.r = r
        self.dv = dv % (2*np.pi)    # keep inside 0:360

        self.boundary = boundary

    def dbins(self, nb, window=[-0.5,2]):
        """
        :param nb: number of bins
        :param window: degress of window around predicted min/max deviation
        """
        d = self.dv 
        dmin = d.min() + window[0]*deg
        dmax = d.max() + window[1]*deg
        return np.linspace(dmin,dmax, nb)


    def refractive_index(self): 
        """
        Plateau in refractive index below 330nm for Glass, 
        edge of data artifact
        """
        wd = np.arange(80,820,10)
        nd = self.boundary.imat.refractive_index(wd)  

        plt.plot(wd, nd)

        return wd, nd







class Rainbow(object):
    """
    Position indices for first k=1 rainbow::
  
          
          4        
                 3T----\
                 /      \
          0-----1T      2R
                  \____/


    In general have k internal reflections BR sandwiched between
    the transmits BT.
    """
    def __init__(self, evt, boundary, k=1, side=[0,0,1]):
        """
        :param side: vector perpendicular to incident rays, 
                     used to define the side for 0:180,180:360 splitting
        """

        if k == -1:
            ssel = "BR SA"  
        else:
            ssel = "BT " + "BR " * k + "BT SA"  

        sel = Selection(evt, ssel) 
        cie = CIE(colorspace="sRGB/D65", whitepoint=evt.whitepoint)

        w = sel.wl

        p0 = sel.recpost(0)[:,:3]
        p1 = sel.recpost(1)[:,:3]
        p_in = p1 - p0  

        assert len(w) == len(p0) == len(p1)

        # assuming axial incidence towards +X
        assert np.all( p_in[:,1] == 0)
        assert np.all( p_in[:,2] == 0)


        pp = sel.recpost(3+k-1)[:,:3]
        pl = sel.recpost(3+k)[:,:3]
        p_out = pl - pp

        side = np.tile(side, len(p_in)).reshape(-1,3)
        assert np.all(np.sum(side*p_in, axis=1) == 0.), "side vector must be perpendicular to incident vectors"
        cside = costheta_(p_out, side)

        cdv = costheta_(p_in, p_out)
        dv = np.piecewise( cdv, [cside>=0, cside<0], [np.arccos,lambda _:2*np.pi - np.arccos(_)])  


        xbow = XRainbow(w, boundary, k=k )

        self.p0 = p0
        self.pl = pl
        self.p_in = p_in
        self.p_out = p_out
        self.cside = cside
        self.evt = evt
        self.cie = cie
        self.xbow = xbow
        self.ssel = ssel
        self.sel = sel
        self.w = w
        self.cdv = cdv
        self.dv = dv
        self.pos = pl
        self.k = k

    def __repr__(self):
        return "Rainbow k=%s %s " % (self.k, self.ssel)
        
    def deviation_plot(self):
        """
        (with mono 500nm) sharp spike at 138 deg  (180-138=42) 
        (nothing below that but extending up to 170) 
        """
        dv = self.dv 
        plt.hist(dv*180./np.pi, bins=100)

    def deviation_vs_wavelength(self):
        """
        kink in deviation against wavelength at 330 nm 
        for both simulated and expected (with Glass)
        suggests its an issue with the refractive index  
    
        YEP: see same shape in refractive index alone, 
        plt.close(); plt.hist2d( n, w, bins=100)
        """
        dv = self.dv 
        w = self.sel.w

        plt.hist2d(dv*180./np.pi, w, bins=100)


    def cf_deviation(self, xbow):
        """
        spikes at zero, but asymmetric tail off to -13 degrees (GlassSchottF2)
        very big zero spike, but long asymmetruc tail off to -40 degress (MainH2OHale)
        same behavior with monochromatic 500nm
        """
        dv = self.dv
        xv = xbow.dv

        plt.hist((xv - dv)*180./np.pi, bins=100)  

    def cieplot_1d(self, b=None, nb=100, ntile=50, norm=2):

        w = self.w
        d = self.dv/deg
        db = self.xbow.dbins(nb)/deg

        if b is None:b = w > 0

        hRGB_raw, hXYZ_raw, bx= self.cie.hist1d(w[b],d[b], db, norm=norm)

        hRGB_1d = np.clip(hRGB_raw, 0, 1)

        hRGB = np.tile(hRGB_1d, ntile ).reshape(-1,ntile,3)

        extent = [0,2,bx[0],bx[-1]] 

        ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1, aspect='auto')

        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        for x in bx[::20]:
            ax.annotate("%3d" % x, xy=(0.5, x), color='white')

        return hRGB

    def hist_1d(self, b=None, nb=100, ntile=50):

        d = self.dv/deg
        db = self.xbow.dbins(nb)/deg
        if b is None:b = d > 0

        h, hx = np.histogram(d[b],bins=db)   
        extent = [0,1,hx[0],hx[-1]] 
        ht = np.repeat(h,ntile).reshape(-1, ntile)
        im = ax.matshow(ht, origin="lower", extent=extent, alpha=1, aspect='auto')
        fig.colorbar(im)
        return ht

    def plot_1d(self, b=None, nb=100):

        d = self.dv/deg
        db = self.xbow.dbins(nb)/deg
        if b is None:b = d > 0

        plt.hist(d[b], bins=db)

        return d[b]

    def cieplot_2d(self, b=None, nb=100, norm=2):

        w = self.w
        x = self.pos[:,0]
        y = self.pos[:,1]
        z = self.pos[:,2]

        yb = np.linspace(y.min(), y.max(), nb)
        zb = np.linspace(z.min(), z.max(), nb)

        r = np.sqrt(y*y + z*z)
        if b is None:b = w > 0

        hRGB_raw, hXYZ_raw, extent = self.cie.hist2d(w[b],y[b],z[b], yb, zb, norm=norm)
        hRGB = np.clip(hRGB_raw, 0, 1)
        ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1)

    def hist_2d(self, b=None, nb=100):

        w = self.w
        x = self.pos[:,0]
        y = self.pos[:,1]
        z = self.pos[:,2]
        if b is None:b = w > 0

        yb = np.linspace(y.min(), y.max(), nb)
        zb = np.linspace(z.min(), z.max(), nb)

        _,_,_,im = ax.hist2d(y[b],z[b], bins=[yb,zb]) 
        fig.colorbar(im)

    def pos_3d(self, b=None):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.pos[:,0]
        y = self.pos[:,1]
        z = self.pos[:,2]
        r = np.sqrt(y*y + z*z)

        if b is None:b = r > -1

        ax.scatter(x[b], y[b], z[b])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()







if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    plt.ion()
    plt.close()

    boundary = Boundary("Vacuum///MainH2OHale")


    # created with ggv-;ggv-rainbow green    etc..
    # huh the blue is coming out magenta 
    # (maximal Z at 
    white, red, green, blue = "1","2","3","4"

    evt = Evt(tag=white, det="rainbow")


    bows = {}

    # direct reflection for white point checking  
    # but it aint peaked enough ...
    #bows[-1] = Rainbow(evt, boundary, k=-1)  

    nk = 6  # restricted by bounce max, record max of the simulation
    nk = 1

    for k in range(1,nk+1):
        bows[k] = Rainbow(evt, boundary, k=k) 

    bow = bows[1]

    w = bow.w 
    dv = bow.dv
    x = bow.pos[:,0]
    y = bow.pos[:,1]
    z = bow.pos[:,2]
    r = np.sqrt(y*y + z*z)

    xbow = bow.xbow 
    n = xbow.n
    xv = xbow.dv


if 1:
    fig = plt.figure()
    for i,k in enumerate(range(1,nk+1)):
        ax = fig.add_subplot(1,nk,i+1)
        bows[k].cieplot_1d(norm=5)

if 0:
    fig = plt.figure()
    for k in range(1,nk+1):
        ax = fig.add_subplot(1,nk,k)
        bows[k].hist_1d()

if 0:
    fig = plt.figure()
    for k in range(1,nk+1):
        ax = fig.add_subplot(1,nk,k)
        bows[k].plot_1d()



if 0:
    fig = plt.figure()

    ax = fig.add_subplot(131)
    bow.cieplot_2d()

    ax = fig.add_subplot(132)
    bow.hist_2d()

    #ax = fig.add_subplot(133)
    #bow.plot_2d()







