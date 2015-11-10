#!/usr/bin/env python
"""
Photon Record positions and time need domains to uncompress.::

    125     rbuffer[record_offset+0] = make_short4(    // 4*int16 = 64 bits 
    126                     shortnorm(p.position.x, center_extent.x, center_extent.w),
    127                     shortnorm(p.position.y, center_extent.y, center_extent.w),
    128                     shortnorm(p.position.z, center_extent.z, center_extent.w),
    129                     shortnorm(p.time      , time_domain.x  , time_domain.y  )
    130                     ); 
    131     

See also

* RecordsNPY::unshortnorm

::

    In [33]: fdom
    Out[33]: 
    array([[[   0.,    0.,    0.,  500.]],      # m_composition->getDomainCenterExtent();

           [[   0.,  200.,    7.,    0.]],      # m_composition->getTimeDomain();

           [[  60.,  810.,   20.,  750.]]], dtype=float32)

    In [35]: idom
    Out[35]: array([[[9, 0, 0, 0]]], dtype=int32)


"""
import numpy as np
import os, logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 
from env.numerics.npy.fresnel import Fresnel


idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )

np.set_printoptions(suppress=True, precision=3)



def scatter3d(fig,  xyz): 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])

def histo(fig,  vals): 
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=91,range=[0,90])

def theta(xyz):
    r = np.linalg.norm(xyz, ord=2, axis=1)
    z = xyz[:,2]
    th = np.arccos(z/r)*180./np.pi
    return th


class Reflect(object):
    def __init__(self, tag="1", src="torch", target=[0,0,300]):
        ox = load_("ox"+src,tag) 
        rx = load_("rx"+src,tag) 
        ph = load_("ph"+src,tag)
        pass
        # select photons and records from two categories
        seqhis = ph[:,0,0]
        sqi = seqhis_int("TORCH BR SA")
        sqj = seqhis_int("TORCH BR AB")
        oxs = np.logical_or(seqhis == sqi, seqhis == sqj)       
        rxs = np.repeat(oxs, 10)
        sf = rx[rxs].reshape(-1,10,2,4)

        # decompress 3 positions, start/refl-point/end
        fdom = np.load(idp_("OPropagatorF.npy"))
        center = fdom[0,0,:3] 
        extent = fdom[0,0,3] 
        p0 = sf[:,0,0,:3].astype(np.float32)*extent/32767.0 + center - target
        p1 = sf[:,1,0,:3].astype(np.float32)*extent/32767.0 + center - target 
        p2 = sf[:,2,0,:3].astype(np.float32)*extent/32767.0 + center - target 

        misser = p1[:,2] != p1[0,2]
        th0 = theta(p0[np.logical_not(misser)])
        th2 = theta(p2[np.logical_not(misser)])  
        th = th0
        # th0 and th2 are incident and reflected angles, matching very well

        self.misser = misser
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.th0 = th0
        self.th2 = th2
        self.th = th

    def plot_misser(self):
        """
        6 percent, 3d plot shows on sides of inner block 
        hmm the triangulated face has a crack down the diagonal
        may getting thru the gap ? 
        so their reflection not in intended place

        distinct structure, with islands, somehow inversely related to the 
        position of cracks in geometry
        """
        misser = self.misser
        p0m = self.p0[misser]  # destined to miss
        p1m = self.p1[misser]   

        fig = plt.figure()
        scatter3d(fig, p0m)  
        fig.show()



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)


    fig = plt.figure()

    dom = np.linspace(0,90,250)
    nlhs = 10   # S/P/U all same at near normal incidence

    fr = Fresnel(dom=dom)
    s = Reflect(tag="1")
    p = Reflect(tag="2")

    fs, bs = np.histogram(s.th, bins=dom)
    fp, bp = np.histogram(p.th, bins=dom)


    ax = fig.add_subplot(111)

    s_lhs = np.average(fr.a_ppol[:nlhs])/np.average(s.th[:nlhs])
    plt.plot( bs[:-1], fs*s_lhs, drawstyle="steps", label="s")

    p_lhs = np.average(fr.a_ppol[:nlhs])/np.average(p.th[:nlhs])
    plt.plot( bp[:-1], fp*p_lhs, drawstyle="steps", label="p")

    fr.pl()
    fr.angles()

    legend = ax.legend(loc='upper left', shadow=True) 

    fig.show()


