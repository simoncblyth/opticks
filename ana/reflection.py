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

from opticks.ana.base import opticks_environment
from opticks.ana.nbase import count_unique
from opticks.ana.types import *

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from opticks.ana.evt import Evt, costheta_
from opticks.ana.ana import Rat, theta, recpos_plot, angle_plot
from opticks.ana.geometry import Boundary
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

    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    es = Evt(tag="1", label="S", det="reflect")
    ep = Evt(tag="2", label="P", det="reflect")

    normal = [0,0,-1]
    source = [0,0,-200] 

    #fig = plt.figure()
    #recpos_plot(fig, [es,ep], origin=source)

    #fig = plt.figure()
    #angle_plot(fig, [es,ep], axis=normal, origin=source)

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

    oneplot(fr,rp,log_=False)
    #oneplot(fr,rp,log_=True)
    #twoplot(fr,rp)



"""

Check starting position and polarization::

    pos = es.rpost_(0)[:,:3]
    pol = es.rpol_(0)

    In [9]:  pos[:,0].min()  A(-99.9786370433668)
    In [10]: pos[:,0].max()  A(100.0091555528428)
    In [12]: pos[:,1].min()  A(-99.9786370433668)
    In [11]: pos[:,1].max()  A(99.9786370433668)
    In [13]: pos[:,2].min()  A(-299.9969481490524)
    In [14]: pos[:,2].max()  A(-200.0183111056856)

    ## sphere radius 100 is hemi-sphered thru center at (0,0,-200) 
    

    Hmm nothing in z 

    In [16]: pol
    Out[16]: 
    array([[ 0.15 ,  0.37 ,  0.   ],
           [ 0.031,  0.118,  0.   ],
           [ 1.   ,  0.   ,  0.   ],
           ..., 
           [-0.646, -0.63 ,  0.   ],
           [-0.276,  0.756,  0.   ],
           [ 0.094,  0.055,  0.   ]])

::

    In [27]: pos[:5]
    Out[27]: 
    A()sliced
    A([[ -36.744,   14.954, -291.787],
           [ -11.994,    3.052, -299.234],
           [  -0.122,   99.826, -206.122],
           [  -4.883,    0.153, -299.875],
           [  11.078,    3.845, -299.295]])

    In [28]: pol[:5]
    Out[28]: 
    array([[ 0.15 ,  0.37 ,  0.   ],     ## spherePosition.y, -spherePosition.x, 0 
           [ 0.031,  0.118,  0.   ],
           [ 1.   ,  0.   ,  0.   ],
           [ 0.   ,  0.047,  0.   ],
           [ 0.039, -0.11 ,  0.   ]])

    In [29]: pos[:5]/100.  ## divide by radius to match normalized spherePosition 
    Out[29]: 
    A()sliced
    A([[-0.367,  0.15 , -2.918],
           [-0.12 ,  0.031, -2.992],
           [-0.001,  0.998, -2.061],
           [-0.049,  0.002, -2.999],
           [ 0.111,  0.038, -2.993]])



::

    464           float sinTheta, cosTheta;
    465           if(ts.mode & M_FLAT_COSTHETA )
    466           {
    467               cosTheta = 1.f - 2.0f*u1 ;
    468               sinTheta = sqrtf( 1.0f - cosTheta*cosTheta );
    469           }
    470           else if( ts.mode & M_FLAT_THETA )
    471           {
    472               sincosf(1.f*M_PIf*u1,&sinTheta,&cosTheta);
    473           }
    474 
    475           float3 spherePosition = make_float3( sinTheta*cosPhi, sinTheta*sinPhi, cosTheta );
    476 
    477           p.position = ts.x0 + radius*spherePosition ;
    478 
    479           p.direction = -spherePosition  ;
    480 
    481           p.polarization = ts.mode & M_SPOL ?
    482                                                make_float3(spherePosition.y, -spherePosition.x , 0.f )
    483                                             :
    484                                                make_float3(-spherePosition.x, -spherePosition.y , 0.f )
    485                                             ;




"""
