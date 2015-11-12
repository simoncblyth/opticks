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

from env.numerics.npy.ana import Evt, Selection, Rat, theta
from env.numerics.npy.fresnel import Fresnel

np.set_printoptions(suppress=True, precision=3)

def scatter3d(fig,  xyz): 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])

def histo(fig,  vals): 
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=91,range=[0,90])

rat_ = lambda n,d:float(len(n))/float(len(d))


class Reflect(object):
    def __init__(self, evt, focus):
        """
        :param evt:
        :param focus: coordinates of reflection point
        """

        al = Selection(evt)
        p0a = al.recpos(0) - focus
        tha = theta(p0a) 
        # initial position with no selection applied

        br = Selection(evt,"BR SA","BR AB")

        p0 = br.recpos(0) - focus
        p1 = br.recpos(1) - focus
        p2 = br.recpos(2) - focus
        # 3 positions, start/refl-point/end for single reflection selection

        miss = p1[:,2] != p1[0,2]
        # missers confirmed to be a triangulation crack effect
        # avoid by not targetting cracks with millions of photons whilst doing reflection tests
        # ...get **ZERO** missers when avoid cracks ...

        th0 = theta(p0[np.logical_not(miss)])
        th2 = theta(p2[np.logical_not(miss)])  

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

        self.focus = focus
        self.al = al 
        self.br = br 
        self.p0a = p0a
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.miss = miss
        self.th0 = th0
        self.th2 = th2
        self.th = th
        self.tha = tha
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
    def __init__(self, fr, focus):

        e1 = Evt(tag="1", label="S")
        e2 = Evt(tag="2", label="P")

        s = Reflect(e1, focus=focus)
        p = Reflect(e2, focus=focus)
        pass
        self.fr = fr
        self.focus = focus
        self.dom = fr.dom
        self.e1 = e1
        self.e2 = e2
        self.s = s
        self.p = p

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
        msize = self.e1.msize()
        return "npy-/reflection.py %3.1fM: %s : %s " % (msize,repr(self.focus),self.fr.title())

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
    focus = [10,0,300]

    fr = Fresnel(m1="Vacuum", m2="Pyrex", wavelength=500., dom=dom) 
    rp = ReflectionPlot(fr, focus)

    oneplot(fr,rp,log_=False)
    #oneplot(fr,rp,log_=True)
    #twoplot(fr,rp)


