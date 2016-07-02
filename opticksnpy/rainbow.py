#!/usr/bin/env python
"""
Catching Rainbows
===================

Using derivations from: Jearl D. Walker
"Multiple rainbows from single drops of water and other liquids",  

* http://www.patarnott.com/atms749/pdf/MultipleRainbowsSingleDrops.pdf

Alexanders dark band, between the 1st and 2nd bows 
(due to no rays below min deviation for each bow)

Polarization Check
-------------------

Plane of incidence defined by initial direction vector 
(a constant vector) and the surface normal at point of incidence, 
which will be different for every intersection point. 

Thus need to specially prepare the polarizations in order to
arrange S-polarized incident light. Basically need to 
calculate surface normal for all points of sphere.

S-polarized :  perpendicular to the plane of incidence


Rendering Spectra
-------------------

Comparison of several approaches to handling out of gamut spectral colors

* http://www-rohan.sdsu.edu/~aty/explain/optics/rendering.html

Collection of color science links

* http://www.midnightkite.com/color.html


Rainbow Calculations
----------------------

The mathematical physics of rainbows and glories, John A. Adam

* http://ww2.odu.edu/~jadam/docs/rainbow_glory_review.pdf



Optics of a water drop
------------------------

* http://www.philiplaven.com/index1.html

Fig 4, Provides relative intensities of S/P-polarizations 
at each step for primary bow.  


Thru multiple Relect/Transmit : is S/P polarization retained ?
------------------------------------------------------------------

S/P polarization is defined with respect to the surface normal 
at the point if incidence.  

Every reflection/transmission happens in the same plane, so that 
suggests  



Maybe the assumption of constant polarization
state is in fact a valid one ?  

* does this depend on the geometry 





"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from env.numerics.npy.ana import Evt, Selection, costheta_, cross_
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.fresnel import fresnel_factor
from env.numerics.npy.cie  import CIE
from env.numerics.npy.droplet  import Droplet
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
        self.boundary = boundary
        self.droplet = Droplet(boundary)
        self.w = w  
        self.k = k

        redblue = np.array([780., 380.])
        self.dvr = self.droplet.deviation_angle(redblue, k)
        self.dv = self.droplet.deviation_angle(w, k)


    def dbins(self, nb, window=[-0.5,0.5]):
        """
        :param nb: number of bins
        :param window: degress of window around predicted min/max deviation
        """
        d = self.dvr 
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
            assert 0
            ssel = None
            sel = Selection(evt) 
        else:
            ssel = "BT " + "BR " * k + "BT "  
            sel = Selection(evt, ssel) 
        pass


        cie = CIE(colorspace="sRGB/D65", whitepoint=evt.whitepoint)

        w = sel.wl
        #b = np.logical_and( w > 380., w < 780. )   # visible range
        b = np.logical_and( w > 200., w < 800. )  

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

        self.b = b
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




class Rainbows(object):
    """
    # max rainbow index restricted by bounce max, record max of the simulation
    """
    def __init__(self, evt, boundary, nk=6):
        bows = {}
        for k in range(1,nk+1):
            bows[k] = Rainbow(evt, boundary, k=k) 

        self.nk = nk 
        self.bows = bows
        self.evt = evt

    def keys(self):
        return self.bows.keys()
 
    def __getitem__(self, key):
        return self.bows[key]

    def selection_counts(self):
        npho = len(self.evt.wavelength) 
        log.info("evt %s : %s " % (repr(self.evt),npho))
        for key in sorted(self.keys()):
            bow = self[key]
            log.info("bow %4s : %6d %4.3f " % (key, len(bow.w), float(len(bow.w))/float(npho) ))




class XFrac(object):
    """
    S-pol/P-pol (polarized perperndicular/parallel to plane of incidence) intensity fraction
    """
    def __init__(self, n, k=np.arange(1,6)):

        i = np.arccos( np.sqrt((n*n - 1.)/(k*(k+2.)) ))  # bow angle
        r = np.arcsin( np.sin(i)/n )                    

        # rainbow paper 
        #     Jearl D. Walker p426, demo that ek1 is indep of n 
        #
        #     derivations assume that the S/P polarization will stay the same 
        #     across all the reflections, that seems surprising 
        # 
        #     swapped the sin and tan for S/P factors
        # 

    
        # perpendicular to plane of incidence, S-pol 
        fs = np.power( np.sin(i-r)/np.sin(i+r) , 2 )
        ts = 1 - fs 
        s = ts*ts*np.power(fs, k)

        # parallel to plane of incidence, P-pol 
        fp = np.power( np.tan(i-r)/np.tan(i+r) , 2 )
        tp = 1 - fp
        p = tp*tp*np.power(fp, k)   


        kk = np.sqrt( k*k + k + 1 )
        qq = (kk - 1)/(kk + 1)
        pq = np.power((1-qq*qq),2)*np.power(qq, 2*k)      

        self.i = i
        self.r = r

        self.fp = fp
        self.tp = tp
        self.p = p

        self.fs = fs
        self.ts = ts
        self.s = s

        self.t = s + p 


        self.kk = kk 
        self.qq = qq 
        self.pq = pq






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

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

    w_evt = Evt(tag=white, det="rainbow", label="3M")
    p_evt = Evt(tag=ppol, det="rainbow", label="P")
    s_evt = Evt(tag=spol, det="rainbow", label="S")

    n = boundary.imat.refractive_index(w_evt.wavelength) 

    navg = (n.min() + n.max())/2.

    nred = 1.331
    xfa = XFrac(nred)

    w_bows = Rainbows(w_evt, boundary, nk=6)
    p_bows = Rainbows(p_evt, boundary, nk=6)
    s_bows = Rainbows(s_evt, boundary, nk=6)

    w_bows.selection_counts()
    p_bows.selection_counts()
    s_bows.selection_counts()


    bow = s_bows[1]

    w = bow.w 
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







