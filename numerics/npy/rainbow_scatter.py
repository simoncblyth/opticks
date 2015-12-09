#!/usr/bin/env python
"""
"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from env.numerics.npy.ana import Evt, Selection, costheta_, cross_
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.fresnel import fresnel_factor
deg = np.pi/180.






   
def scattering_angle_plot(pmax=4, n=1.33):
    idom = np.arange(0,90,1)*deg 
    fig = plt.figure()
    fig.suptitle("Scattering angle for different orders, 0:R, 1:T, 2:TRT, 3:TRRT ")
    ax = fig.add_subplot(1,1,1)
    for i,p in enumerate(range(pmax)):
        thp = scattering_angle(idom, p, n=n)
        msk = ~np.isnan(thp)
        ax.plot( idom[msk]/deg, thp[msk]/deg, label=p)
    pass
    ax.legend()


def intensity(i, k=1, n=1.33):
    """
    This goes to infinity at the bow angles 
    """
    r = np.arcsin( np.sin(i)/n )                    
    dthdi = 2.-2*(k+1)*np.tan(r)/np.tan(i)
    return 1./dthdi


def intensity_plot(k=1, n=1.33):
    """
    Need intensity as function of scattering angle summing over all bows 
    (and individually per bow and for reflection and transmission too)
    to compare against the unselected deviation plot

    Hmm this depends on the distribution of incident angles ...
    """
    dom = np.linspace(0,360,361)
    it = intensity(dom*deg, k=k, n=n )
    msk = ~np.isnan(it)
    plt.plot( dom[msk], it[msk] )




class Scatter(object):
    """
    View a a general scattering process, without assuming particular bow angle

    Comparison of the various ways of calculating rainbows: Descartes, Newton, Mie, Debybe, Airy

    * http://www.philiplaven.com/p8e.html


    :param p: type of scatter

    p=0 
          reflection from the exterior of the sphere;
    p=1
          transmission through the sphere
    p=2
          one internal reflection
    p=3
          two internal reflections

    p=0     2i - 180
    p=1     2i - 2r
    p > 1   2i - 2pr + (p-1)180

    """

    def __init__(self, p, n1=1, n2=1.33257):

        k = p - 1
        if p == 0:
            seq = "BR "
        elif p == 1:
            seq = "BT BT "
        elif p > 1:
            seq = "BT " + "BR " * k + "BT "  
        else:
            assert 0 

        seq += "SA "

        self.p = p
        self.k = k
        self.n1 = n1
        self.n2 = n2
        self.n  = n2/n1
        self.seq = seq

    def __repr__(self):
        extra = " BOW k=%d " % self.k if self.p > 1 else ""
        return "Scatter p=%d %s  %s" % (self.p, self.seq, extra)

    def theta(self, i):
        """
        :param i: incident angle arising from beam characteristic, and orientation to the sphere
        :return scattering angle:        

        For a half illuminated drop the incident angle will vary from
        zero at normal incidence to pi/2 at tangential incidence
        """
        n = self.n
        p = self.p
        k = self.k
 
        i = np.asarray(i)
        r = np.arcsin( np.sin(i)/n )

        #th = np.piecewise( i, [p==0,p==1,p>1], [np.pi - 2*i, 2*(i-r), k*np.pi + 2*i - 2*r*(k+1) ])  
        # should be lambda funcs to get this to work 
    
        if p == 0:
           th = 2*i - np.pi 
        elif p == 1:
           th = 2*(i-r)
        elif p > 1:
           th = k*np.pi + 2*i - 2*r*(k+1)
        else:
           assert 0    

        # maybe discrep between angle range convention ?
        return th % (2.*np.pi)

    def dthetadi(self, i):
        """
        This goes to infinity at the bow angles 
        """
        n = self.n
        p = self.p
        k = self.k
        i = np.asarray(i)
        r = np.arcsin( np.sin(i)/n )                    

        if p == 0:
            dthdi = 2.*np.ones(len(i))
        elif p == 1:
            dthdi = 2.-2*np.tan(r)/np.tan(i)
        elif p > 1:
            dthdi = 2.-2*(k+1)*np.tan(r)/np.tan(i)
        else:
            assert 0
        return dthdi

    def geometry(self, i):
        th = self.theta(i) 
        geom = 0.5*np.sin(2*i)/np.sin(th)
        return geom


    def intensity(self, i):
        """
        Particular incident angle yields a particular scatter angle theta, 
        BUT how much is scattered in a particular theta bin relative to the
        total ?
        """
        pass
        th = self.theta(i)
        dthdi = self.dthetadi(i)
        geom = self.geometry(i)
        s = fresnel_factor(self.seq, i, self.n1, self.n2, spol=True)
        p = fresnel_factor(self.seq, i, self.n1, self.n2, spol=False)

        s_inten = s*geom/dthdi 
        p_inten = p*geom/dthdi 

        return th, s_inten, p_inten


    @classmethod
    def combined_intensity(cls, pp=[1,2,3],  n1=1, n2=1.33257):
        idom = np.arange(0,90,.1)*deg 
        c_th = [] 
        c_si = []
        c_pi = []
        for p in pp:
            sc = Scatter(p, n1=n1, n2=n2)
            th,si,pi = sc.intensity(idom)
            c_th.append(th)
            c_si.append(si)
            c_pi.append(pi)
        pass
        cth = np.hstack(c_th)
        csi = np.hstack(c_si)
        cpi = np.hstack(c_pi)

        return cth, csi, cpi

    @classmethod
    def combined_intensity_plot(cls, pp=[1,2,3,4], log_=True, ylim=None, flip=False, scale=1, mask=False):
        tsp = cls.combined_intensity(pp)
        cls._intensity_plot(tsp, log_=log_, ylim=ylim, flip=flip, scale=scale, mask=mask)
            
    @classmethod
    def _intensity_plot(cls, tsp, log_=True, ylim=None, flip=False, scale=1, mask=False):

        th, s_inten, p_inten = tsp
        if flip:
           s_inten = -s_inten 
           p_inten = -p_inten 

        if mask:
           msk = s_inten > 0
        else:
            msk = np.tile(True, len(th)) 

        ax = plt.gca()
        ax.plot( th[msk]/deg, s_inten[msk]*scale )
        ax.plot( th[msk]/deg, p_inten[msk]*scale )
 
        if ylim:
            ax.set_ylim(ylim)
        if log_:
            ax.set_yscale('log')

    def intensity_plot(self, i, log_=True, ylim=None, flip=False, scale=1, mask=False):
        """
        Lots of negative lobes
        """
        tsp = self.intensity(i)
        self._intensity_plot(tsp, log_=log_, ylim=ylim, flip=flip, scale=scale, mask=mask)

    def table(self, i):
        th = self.theta(i)
        dthdi = self.dthetadi(i)
        geom = self.geometry(i)
        s = fresnel_factor(self.seq, i, self.n1, self.n2, spol=True)
        p = fresnel_factor(self.seq, i, self.n1, self.n2, spol=False)

        s_inten = s*geom/dthdi 
        p_inten = p*geom/dthdi 

        log.info("%s [i/deg,s,p,s+p,th/deg,dthdi, geom, s_inten, p_inten]", self)
        return np.dstack([i/deg,s,p,s+p,th/deg,dthdi, geom, s_inten, p_inten])



def scatter_plot(pevt, sevt):
    s_dv0 = sevt.deviation_angle()
    p_dv0 = pevt.deviation_angle()
    db = np.arange(0,360,1)

    ax = fig.add_subplot(1,1,1)

    for i,d in enumerate([s_dv0/deg, p_dv0/deg]):
        ax.set_xlim(0,360)
        ax.set_ylim(1,1e5)
        cnt, bns, ptc = ax.hist(d, bins=db,  log=True, histtype='step')
    pass

        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    #intensity_plot() 
    #scattering_angle_plot() 

    
    step = .1
    #step = 10
    idom = np.arange(0,90,step)*deg 
    ylim = None

if 0:
    fig = plt.figure()
    ylim = [1e-6, 1e2]
    ax = fig.add_subplot(1,1,1)

if 0:
    fig = plt.figure()
    fig.suptitle("Deviation angles without selection")

    spol,ppol = "5", "6"
    p_evt = Evt(tag=ppol, det="rainbow", label="P")
    s_evt = Evt(tag=spol, det="rainbow", label="S")

    ax = fig.add_subplot(1,1,1)
    scatter_plot(p_evt, s_evt)
    Scatter.combined_intensity_plot([1,2,3,4,5], ylim=ylim, scale=5e4, flip=False )

    #ax = fig.add_subplot(2,1,2)
    #scatter_plot(p_evt, s_evt)
    #Scatter.combined_intensity_plot([1,2,3,4,5], ylim=ylim, scale=5e4, flip=True )


if 1:
    fig = plt.figure()
    fig.suptitle("Scatter contribs from modes 0:3")
    for i, p in enumerate(range(4)):
        sc = Scatter(p)
        print sc.table(idom)

        ax = fig.add_subplot(2,4,i+1)
        sc.intensity_plot(idom, ylim=None, flip=False, log_=False, mask=True)

        ax = fig.add_subplot(2,4,i+1+4)
        sc.intensity_plot(idom, ylim=ylim, flip=False, log_=True, mask=True)

        #ax = fig.add_subplot(3,4,i+1+8)
        #sc.intensity_plot(idom, ylim=ylim, flip=True, log_=True, mask=True)


if 0:
    #
    # as theta comes in from 180 both s and p are negative, they both go thru the 
    # discontinuiy at the bow angle before theta increases thru the positive lobe
    # ... the lobes are not symmetric 
    # 
    fig = plt.figure()
    fig.suptitle("Scatter intensity for bow 1")
    sc = Scatter(2)
    print sc.table(idom)
    ax = fig.add_subplot(1,1,1)
    idom = np.arange(0,90,1)*deg 
    tsp = sc.intensity(idom)
    Scatter._intensity_plot(tsp, ylim=None, flip=False, log_=True, mask=True)
    #ax.set_ylim([1e-4,10])
    #ax.set_yscale('log')
    d = np.dstack(tsp)
    print d 

    d[0,:,0] /= deg



