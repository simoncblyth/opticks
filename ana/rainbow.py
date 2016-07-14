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
#from opticks.ana.ana import Selection, cross_
#from opticks.ana.ana import cross_
from opticks.ana.boundary import Boundary   
from opticks.ana.fresnel import fresnel_factor
from opticks.ana.cie  import CIE
from opticks.ana.xrainbow import XRainbow

deg = np.pi/180.


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

    @classmethod
    def selection_string(cls, k):
        if k == -1:
            ssel = None
        else:
            ssel = "TO " + "BT " + "BR " * k + "BT " + "SA " 
        pass
        return ssel 

    @classmethod
    def load(cls, devt, k):
        ssel = cls.selection_string(k)
        d = devt.copy()
        d["seqs"] = [ssel] 
        sel = Evt(**d)
        return sel


    def __init__(self, devt, boundary, k=1, side=[1,0,0]):
        """
        :param side: vector perpendicular to incident rays, 
                     used to define the side for 0:180,180:360 splitting
        """
        all_ = devt.copy()
        evt = Evt(**all_) 

        ssel = self.selection_string(k)
        seqs = [ssel]
 
        s = devt.copy()
        s["seqs"] = seqs 
        sel = Evt(**s)

        #cie = CIE(colorspace="sRGB/D65", whitepoint=sel.whitepoint)
        cie = None

        w = sel.wl
        #b = np.logical_and( w > 380., w < 780. )   # visible range
        b = np.logical_and( w > 200., w < 800. )  

        p0 = sel.rpost_(0)[:,:3]
        p1 = sel.rpost_(1)[:,:3]
        p_in = p1 - p0  

        assert len(w) == len(p0) == len(p1)

        # incidence direction is now -Z 
        assert np.all( p_in[:,0] == 0)
        assert np.all( p_in[:,1] == 0)

        pp = sel.rpost_(3+k-1)[:,:3]
        pl = sel.rpost_(3+k)[:,:3]
        p_out = pl - pp

        side = np.tile(side, len(p_in)).reshape(-1,3)
        assert np.all(np.sum(side*p_in, axis=1) == 0.), "side vector must be perpendicular to incident vectors"
        cside = costheta_(p_out, side)

        cdv = costheta_(p_in, p_out)
        dv = np.piecewise( cdv, [cside>=0, cside<0], [np.arccos,lambda _:2*np.pi - np.arccos(_)])  


        xbow = XRainbow(w, boundary, k=k )

        self.b = b
        self.p0 = p0
        self.pl = pl
        self.p_in = p_in
        self.p_out = p_out
        self.cside = cside
        self.evt = evt       ## without history selection 
        self.cie = cie
        self.xbow = xbow
        self.ssel = ssel
        self.sel = sel      ## with history selection  
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

    def cieplot_1d(self, b=None, nb=20, ntile=50, norm=2):

        w = self.w
        d = self.dv/deg
        db = self.xbow.dbins(nb)/deg
        dvr = self.xbow.dvr/deg

        log.info("cieplot_1d %s " % str(dvr))

        if b is None:b = np.logical_and( w > 380., w < 780. )   # visible range

        hRGB_raw, hXYZ_raw, bx= self.cie.hist1d(w[b],d[b], db, norm=norm)

        hRGB_1d = np.clip(hRGB_raw, 0, 1)

        hRGB = np.tile(hRGB_1d, ntile ).reshape(-1,ntile,3)

        extent = [0,2,bx[0],bx[-1]] 

        #interpolation = 'none'
        #interpolation = 'mitchell'
        interpolation = 'gaussian'

        ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1, aspect='auto', interpolation=interpolation)

        # http://stackoverflow.com/questions/16829436/overlay-matplotlib-imshow-with-line-plots-that-are-arranged-in-a-grid
        # trick to overlay lines on top of an image
        box = ax._position.bounds
        tmp_ax = fig.add_axes([box[0], box[1], box[2], box[3]])
        tmp_ax.set_axis_off()
        tmp_ax.set_ylim(extent[2], extent[3])
        tmp_ax.set_xlim(extent[0], extent[1])
        tmp_ax.plot( [0.1, 0.1], [dvr[0],dvr[1]], "w-" )

        tmp_ax.annotate( "%3.1f" % dvr[0], xy=(0.2, dvr[0]), color='white')
        tmp_ax.annotate( "%3.1f" % dvr[1], xy=(0.2, dvr[1]), color='white')

        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        #for x in bx[::20]:
        #    ax.annotate("%3d" % x, xy=(0.5, x), color='white')

        return hRGB

    def hist_1d(self, b=None, nb=100, ntile=50):

        d = self.dv/deg
        db = self.xbow.dbins(nb)/deg

        h, hx = np.histogram(d[b],bins=db)   
        extent = [0,1,hx[0],hx[-1]] 
        ht = np.repeat(h,ntile).reshape(-1, ntile)
        im = ax.matshow(ht, origin="lower", extent=extent, alpha=1, aspect='auto')
        fig.colorbar(im)
        return ht

    def plot_1d(self, b=None, nb=20):

        if b is None:b = self.b
        d = self.dv/deg
        db = self.xbow.dbins(nb)/deg
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
        if b is None:b = self.b

        hRGB_raw, hXYZ_raw, extent = self.cie.hist2d(w[b],y[b],z[b], yb, zb, norm=norm)
        hRGB = np.clip(hRGB_raw, 0, 1)
        ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1, interpolation='none')

    def hist_2d(self, b=None, nb=100):

        w = self.w
        x = self.pos[:,0]
        y = self.pos[:,1]
        z = self.pos[:,2]
        if b is None:b = self.b

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

        if b is None:b = self.b

        ax.scatter(x[b], y[b], z[b])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()



class Rainbows(object):
    """
    # max rainbow index restricted by bounce max, record max of the simulation

    Formerly a Selection class acted upon an evt, 
    but have since moved to doing selecton with Evt. 
    So this is structured awkwardly.
    """
    def __init__(self, devt, boundary, nk=6):
        bows = {}
        for k in range(1,nk+1):
            bows[k] = Rainbow(devt, boundary, k=k) 

        self.nk = nk 
        self.bows = bows
        self.devt = devt
        self.evt = self.bows[1].evt

    def keys(self):
        return self.bows.keys()
 
    def __getitem__(self, key):
        return self.bows[key]

    def selection_counts(self):
        npho = len(self.evt.wl) 
        log.info("evt %s : %s " % (repr(self.evt),npho))
        for key in sorted(self.keys()):
            bow = self[key]
            log.info("bow %4s : %6d %4.3f " % (key, len(bow.w), float(len(bow.w))/float(npho) ))






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    plt.ion()
    plt.close()


    boundary = Boundary("Vacuum///MainH2OHale")

    # created with ggv-;ggv-rainbow green    etc..
    # huh the blue is coming out magenta 
    # (maximal Z at 
    white, red, green, blue, spol, ppol = "1","2","3","4","5", "6"

    p_evt = dict(tag=ppol, det="rainbow", label="P")
    s_evt = dict(tag=spol, det="rainbow", label="S")

    nk = 6 
    #nk = 1 

    p_bows = Rainbows(p_evt, boundary, nk=nk)
    s_bows = Rainbows(s_evt, boundary, nk=nk)

    p_bows.selection_counts()
    s_bows.selection_counts()

    bow = s_bows[1]

    #w = bow.w 
    dv = bow.dv
    x = bow.pos[:,0]
    y = bow.pos[:,1]
    z = bow.pos[:,2]
    r = np.sqrt(y*y + z*z)

    #xbow = bow.xbow 
    #n = xbow.n
    #xv = xbow.dv


