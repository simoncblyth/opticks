#!/usr/bin/env python
"""
treflect.py
=======================

See :doc:`../tests/treflect`

Loads S and P polarized events, selects photons that 
have histories : "TO BR SA" or "TO BR AB", 
ie photons that reflect at the first interface and 
are either bulk absorbed (AB) or surface absorbed (SA). 

A ratio of histograms of the angle of incidence (theta) 
for all photons with reflecting photons is compared 
against expectations of the Fresnel formula. 
 


"""
import os, sys, logging
import numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from opticks.ana.base import opticks_main
from opticks.ana.nbase import count_unique
from opticks.ana.evt import Evt, costheta_
from opticks.ana.ana import Rat, theta, recpos_plot, angle_plot
from opticks.ana.boundary import Boundary
from opticks.ana.fresnel import Fresnel

deg = np.pi/180.

np.set_printoptions(suppress=True, precision=3)

rat_ = lambda n,d:float(len(n))/float(len(d))

X,Y,Z,W = 0,1,2,3


class Reflect(object):

    def theta(self, vecs, normal):
        N = len(vecs)
        nrm = np.tile(normal, N).reshape(-1, len(normal))
        ct = costheta_(vecs, nrm)
        th = np.arccos(ct)*180./np.pi
        return th

    def __init__(self, evt, focus, normal):
        """
        :param evt:
        :param focus: coordinates of reflection point

        
        p0a initial position with no selection applied

        """

        al = evt  
        p0a = al.rpost_(0)[:,:W] - focus
        tha = self.theta( p0a, normal ) 

        br = Evt.selection(evt, seqs=["TO BR SA","TO BR AB"], label="TO BR ..")

        p0 = br.rpost_(0)[:,:W] - focus   # start
        p1 = br.rpost_(1)[:,:W] - focus   # refl point
        p2 = br.rpost_(2)[:,:W] - focus   # end

        #miss = p1[:,2] != p1[0,2]   # this assumes a particular geometry 
        # missers confirmed to be a triangulation crack effect
        # avoid by not targetting cracks with millions of photons whilst doing reflection tests
        # ...get **ZERO** missers when avoid cracks ...


        if 0:
            miss = np.tile(False, len(p1))
            self.miss = miss
            msk = ~miss
            p0u = p0[msk]
            p2u = p2[msk]
        else:
            p0u = p0
            p2u = p2


        th0 = self.theta(p0u, normal)
        th2 = self.theta(p2u, normal)
        th = th0

        missrat = Rat(th0,p0,"th0/p0")

        rats = []
        rats.append(Rat(p0,tha,  "p0/tha"))
        rats.append(Rat(th0,tha, "th0/tha"))
        rats.append(missrat)

        msg = " ".join(map(repr,rats))
        log.info("%s : %s " % (repr(evt),msg))


        # th0 and th2 are incident and reflected angles
        # matching very well after exclude miss-ers

        self.evt = evt
        self.focus = focus
        self.normal = normal
        self.al = al 
        self.br = br 
        self.p0a = p0a
        self.tha = tha
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.th0 = th0
        self.th2 = th2
        self.th = th
        self.missfrac = 1. - missrat.r
        self.rats = rats

    def plot_misser(self):
        """
        6 percent, 3d plot shows on sides of inner block 
        hmm the triangulated face has a crack down the diagonal
        may getting thru the gap ? 
        so their reflection not in intended place

        distinct structure, with islands, somehow inversely related to the 
        position of cracks in geometry

        YEP: confirmed, easily avoided 
        by shifting the focus point to avoid cracks
        """
        miss = self.miss
        p0m = self.p0[miss]  # destined to miss
        p1m = self.p1[miss]   

        fig = plt.figure()
        scatter3d(fig, p0m)  
        fig.show()


class ReflectionPlot(object):
    def __init__(self, s, p, fr):
        self.s = s
        self.p = p
        self.fr = fr
        self.dom = fr.dom

    def ratio(self, xsel, xall):
        dom = self.dom
        fa, ba = np.histogram(xall, bins=dom)
        fs, bs = np.histogram(xsel, bins=dom)
        assert np.all( ba == dom )
        assert np.all( bs == dom )
        rat = fs.astype(np.float32)/fa.astype(np.float32)
        return rat 

    def abs_norm(self):
        """
        Absolute normalization using initial "all" distrib with no selection
        """
        dom = self.dom
        s = self.s
        p = self.p

        sr = self.ratio( s.th, s.tha )
        pr = self.ratio( p.th, p.tha )

        plt.plot( dom[:-1], sr, drawstyle="steps", label="s", c="r")
        plt.plot( dom[:-1], pr, drawstyle="steps", label="p", c="b")

    def theory(self):
        fr = self.fr
        if fr is None:
            return  
        fr.pl()
        fr.angles()

    def title(self):
        msize = self.s.evt.msize()
        return "npy-/reflection.py %3.1fM: %s " % (msize,self.fr.title())

    def legend(self, ax, log_=False):
        if log_:
            ax.set_yscale('log')
            loc = 'lower left'
        else:
            loc = 'upper left'
        legend = ax.legend(loc=loc, shadow=True) 

    def plot(self, ax, log_=False):
        self.abs_norm()
        self.theory()
        self.legend(ax, log_=log_) 


def oneplot(fr,rp, log_=False):
    fig = plt.figure()
    plt.title(rp.title())
    ax = fig.add_subplot(111)
    rp.plot(ax,log_=log_) 
    fig.show()

def twoplot(fr,rp):
    fig = plt.figure()
    plt.title(rp.title())
    ax = fig.add_subplot(121)
    rp.plot(ax,log_=False)
    ax = fig.add_subplot(122)
    rp.plot(ax,log_=True)
    fig.show()

def attic():
    transform = "0.500,0.866,0.000,0.000,-0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,-86.603,0.000,0.000,1.000"
    tx = np.fromstring(transform, sep=",").reshape(4,4)
    focus = np.dot([0,0,0,1],tx)[:3]
    normal = np.dot([0,1,0,0],tx)[:3]



if __name__ == '__main__':

    args = opticks_main(det="reflect",stag="1", ptag="2")

    try:
        es = Evt(tag=args.stag, label="S", det=args.det, args=args)
        ep = Evt(tag=args.ptag, label="P", det=args.det, args=args)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc) 

    log.info(" es : %s " % es.brief )
    log.info(" ep : %s " % ep.brief )


    if not (es.valid and ep.valid):
        log.fatal("both es and ep must be valid")
        sys.exit(1)
    pass


    normal = [0,0,-1]
    source = [0,0,-200] 

    plt.ion()
    fig = plt.figure()
    recpos_plot(fig, [es,ep], origin=source)



    fig = plt.figure()
    angle_plot(fig, [es,ep], axis=normal, origin=source)

    swl = es.unique_wavelength()
    pwl = ep.unique_wavelength()
    assert swl == pwl 

    #boundary = Boundary("Vacuum///GlassSchottF2")
    boundary = Boundary("Vacuum///MainH2OHale")
    
    wl = swl
    n1 = boundary.omat.refractive_index(wl)  
    n2 = boundary.imat.refractive_index(wl)  

    dom = np.linspace(0,90,250)
    fr = Fresnel(n1, n2, dom=dom) 

    s = Reflect(es, focus=source, normal=normal)
    p = Reflect(ep, focus=source, normal=normal)

    rp = ReflectionPlot(s, p, fr)

    #oneplot(fr,rp,log_=False)
    oneplot(fr,rp,log_=True)
    #twoplot(fr,rp)




