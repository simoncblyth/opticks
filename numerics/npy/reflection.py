#!/usr/bin/env python
"""
Reflection distrib following BoxInBox::

    ggv-
    ggv-reflect s
    ggv-reflect p

::

    ggv-reflect() 
    { 
        local pol=${1:-s};
        case $pol in 
            s)
                tag=1
            ;;
            p)
                tag=2
            ;;
        esac;
        ggv.sh --test 
               --save 
               --tag $tag 
               --eye 0.5,0.5,0.0 
               --animtimemax 7 
               --testconfig "mode=BoxInBox_dimensions=500,300,0,0_boundary=Rock//perfectAbsorbSurface/Vacuum_boundary=Vacuum///Pyrex_" 
               --torchconfig "polz=${pol}pol_frame=1_type=refltest_source=10,10,300_target=0,0,0_radius=102_zenithazimuth=0,0.5,0,1_material=Vacuum" $*;
    }



                    + [x,y,z-300]
                   /|
                  / |
                 /  | 
                /   |
    -----------+----+--------
        [0,0,300] r 


Focus [0,0,300] in on triangulation crack::

    INFO:__main__:Evt(1,"torch","S") : Rat p0/tha 82600/500000 0.165  Rat th0/tha 77872/500000 0.156  Rat th0/p0 77872/82600 0.943  
    INFO:__main__:Evt(2,"torch","P") : Rat p0/tha 42066/500000 0.084  Rat th0/tha 37998/500000 0.076  Rat th0/p0 37998/42066 0.903  

Focus [10,10,300] still on the (y=x) line crack::

    INFO:__main__:Evt(1,"torch","S") : Rat p0/tha 86783/500000 0.174  Rat th0/tha 83698/500000 0.167  Rat th0/p0 83698/86783 0.964  
    INFO:__main__:Evt(2,"torch","P") : Rat p0/tha 42500/500000 0.085  Rat th0/tha 39846/500000 0.080  Rat th0/p0 39846/42500 0.938  

Focus [10,0,300] avoids the crack, visualizations more physical: no missers, clean cones, near perfect Fresnel match::

    INFO:__main__:Evt(1,"torch","S") : Rat p0/tha 95235/500000 0.190  Rat th0/tha 95235/500000 0.190  Rat th0/p0 95235/95235 1.000  
    INFO:__main__:Evt(2,"torch","P") : Rat p0/tha 43537/500000 0.087  Rat th0/tha 43537/500000 0.087  Rat th0/p0 43537/43537 1.000  

* http://www.ece.rice.edu/~daniel/262/pdf/lecture13.pdf

"""
import os, logging
import numpy as np
from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

from env.numerics.npy.ana import Evt, Selection, Rat, theta, costheta_
from env.numerics.npy.geometry import Boundary
from env.numerics.npy.fresnel import Fresnel

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

        al = Selection(evt)  
        p0a = al.recpost(0)[:,:W] - focus
        tha = self.theta( p0a, normal ) 

        br = Selection(evt,"BR SA","BR AB")

        p0 = br.recpost(0)[:,:W] - focus   # start
        p1 = br.recpost(1)[:,:W] - focus   # refl point
        p2 = br.recpost(2)[:,:W] - focus   # end

        #miss = p1[:,2] != p1[0,2]   # this assumes a particular geometry 
        # missers confirmed to be a triangulation crack effect
        # avoid by not targetting cracks with millions of photons whilst doing reflection tests
        # ...get **ZERO** missers when avoid cracks ...


        miss = np.tile(False, len(p1))

        msk = ~miss
        p0m = p0[msk]
        p2m = p2[msk]

        th0 = self.theta(p0m, normal)
        th2 = self.theta(p2m, normal)

        missrat = Rat(th0,p0,"th0/p0")

        rats = []
        rats.append(Rat(p0,tha,  "p0/tha"))
        rats.append(Rat(th0,tha, "th0/tha"))
        rats.append(missrat)

        msg = " ".join(map(repr,rats))
        log.info("%s : %s " % (repr(evt),msg))


        th = th0
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
        self.miss = miss
        self.th0 = th0
        self.th2 = th2
        self.th = th
        self.missfrac = 1. - missrat.r

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


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    dom = np.linspace(0,90,250)

    #focus = np.array([10,0,300])    

    # ggv-prism the intesect plane is tilted, not horizontal as reflection.py is assuming
    transform = "0.500,0.866,0.000,0.000,-0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,-86.603,0.000,0.000,1.000"
    tx = np.fromstring(transform, sep=",").reshape(4,4)
    focus = np.dot([0,0,0,1],tx)[:3]
    normal = np.dot([0,1,0,0],tx)[:3]

    boundary = Boundary("Vacuum///GlassSchottF2")


    wl = 380 

    n1 = boundary.omat.refractive_index(wl)  
    n2 = boundary.imat.refractive_index(wl)  


    fr = Fresnel(boundary, wavelength=wl, dom=dom) 

    es = Evt(tag="1", label="S", det="prism")
    ep = Evt(tag="2", label="P", det="prism")

    s = Reflect(es, focus=focus, normal=normal)
    p = Reflect(ep, focus=focus, normal=normal)

    rp = ReflectionPlot(s, p, fr)

    #oneplot(fr,rp,log_=False)
    oneplot(fr,rp,log_=True)
    #twoplot(fr,rp)




