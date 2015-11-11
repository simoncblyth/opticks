#!/usr/bin/env python
"""
Reflection distrib following BoxInBox with save::

    ggv-bib --save

    ggv.sh --test \
           --eye 0.5,0.5,0.0 \
           --animtimemax 7 \
           --testconfig "mode=BoxInBox_dimensions=500,300,0,0_boundary=Rock//perfectAbsorbSurface/Vacuum_boundary=Vacuum///Pyrex_" \
           --torchconfig "frame=1_type=invsphere_source=0,0,300_target=0,0,0_radius=102_zenithazimuth=0,0.5,0,1_material=Vacuum" \
            $*


                    + [x,y,z-300]
                   /|
                  / |
                 /  | 
                /   |
    -----------+----+--------
        [0,0,300] r 


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

from env.numerics.npy.ana import Evt, Selection, theta
from env.numerics.npy.fresnel import Fresnel

np.set_printoptions(suppress=True, precision=3)

def scatter3d(fig,  xyz): 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])

def histo(fig,  vals): 
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=91,range=[0,90])


class Reflect(object):
    def __init__(self, evt, target=[0,0,300]):

        al = Selection(evt)
        p0a = al.recpos(0) - target
        # initial position with no selection applied

        br = Selection(evt,"BR SA","BR AB")
        p0 = br.recpos(0) - target
        p1 = br.recpos(1) - target
        p2 = br.recpos(2) - target
        # 3 positions, start/refl-point/end for single reflection selection

        miss = p1[:,2] != p1[0,2]
        # TODO: investigate the missers : maybe triangulation crack effect

        th0 = theta(p0[np.logical_not(miss)])
        th2 = theta(p2[np.logical_not(miss)])  
        th = th0
        # th0 and th2 are incident and reflected angles
        # matching very well after exclude miss-ers
        tha = theta(p0a) 

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

    def plot_misser(self):
        """
        6 percent, 3d plot shows on sides of inner block 
        hmm the triangulated face has a crack down the diagonal
        may getting thru the gap ? 
        so their reflection not in intended place

        distinct structure, with islands, somehow inversely related to the 
        position of cracks in geometry
        """
        miss = self.miss
        p0m = self.p0[miss]  # destined to miss
        p1m = self.p1[miss]   

        fig = plt.figure()
        scatter3d(fig, p0m)  
        fig.show()


class ReflectionPlot(object):
    def __init__(self, dom):
        fr = Fresnel(dom=dom)
        e1 = Evt(tag="1")
        e2 = Evt(tag="2")

        s = Reflect(e1)
        p = Reflect(e2)
        pass
        self.dom = dom
        self.fr = fr
        self.e1 = e1
        self.e2 = e2
        self.s = s
        self.p = p

    def lhs_norm(self, nlhs=10):   
        """
        Left edge normalize, using fact that S/P reflection is 
        same at near normal incidence
        """
        dom = self.dom
        s = self.s
        p = self.p
        fr = self.fr

        fs, bs = np.histogram(s.th, bins=dom)
        fp, bp = np.histogram(p.th, bins=dom)

        s_lhs = np.average(fr.a_ppol[:nlhs])/np.average(s.th[:nlhs])
        p_lhs = np.average(fr.a_ppol[:nlhs])/np.average(p.th[:nlhs])

        plt.plot( bs[:-1], fs*s_lhs, drawstyle="steps", label="s")
        plt.plot( bp[:-1], fp*p_lhs, drawstyle="steps", label="p")

    def ratio(self, xsel, xall):
        dom = self.dom
        fa, ba = np.histogram(xall, bins=dom)
        fs, bs = np.histogram(xsel, bins=dom)
        assert np.all( ba == dom )
        assert np.all( bs == dom )
        rat = fs.astype(np.float32)/fa.astype(np.float32)
        return rat 

    def abs_norm(self, sfudge=1., pfudge=1.):
        """
        Absolute normalization using initial "all" distrib with no selection
        """
        dom = self.dom
        s = self.s
        p = self.p

        sr = self.ratio( s.th, s.tha )
        pr = self.ratio( p.th, p.tha )

        plt.plot( dom[:-1], sr*sfudge, drawstyle="steps", label="s")
        plt.plot( dom[:-1], pr*pfudge, drawstyle="steps", label="p")

    def theory(self):
        fr = self.fr
        fr.pl()
        fr.angles()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    dom = np.linspace(0,90,250)
    rp = ReflectionPlot(dom)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #rp.lhs_norm()
    rp.abs_norm(pfudge=1.0, sfudge=1.0)

    rp.theory()

    legend = ax.legend(loc='upper left', shadow=True) 
    fig.show()


