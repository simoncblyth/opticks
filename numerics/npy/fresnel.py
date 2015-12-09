#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 

np.set_printoptions(suppress=True, precision=3)

logging.basicConfig(level=logging.INFO)


def fresnel(x, n1, n2, spol=True):
    """
    https://en.wikipedia.org/wiki/Fresnel_equations
    """
    cx = np.cos(x)
    sx = np.sin(x) 
    disc = 1. - np.square(n1*sx/n2)
    qdisc = np.sqrt(disc)
    pass
    if spol:
        num = (n1*cx - n2*qdisc)
        den = (n1*cx + n2*qdisc) 
    else:
        num = (n1*qdisc - n2*cx)
        den = (n1*qdisc + n2*cx) 
    pass
    return np.square(num/den) 


def fresnel_factor(seqhis, i, n1, n2, spol=True):
    """
    :param seqhis: history sequence string eg "TO BT BR BT SA "
    :param n1: refractive index of initial material
    :param n2: refractive index of material that is transmitted into 

    Not aiming for generality, only works for simple geometries like raindrops, prisms, lens
    """
    assert len(seqhis) % 3 == 0
    nseq = len(seqhis) / 3
    rx = fresnel(i, n1, n2, spol=spol )
    tx = 1 - rx
    ff = np.ones(len(i))
    for q in range(nseq):
        step = seqhis[q*3:q*3+3]
        #print step
        if step in ("TO ", "SA "):continue
        if step == "BT ":
            ff *= tx
        elif step == "BR ":
            ff *= rx
        else:
            assert 0, step 
        pass
    pass
    return ff




def fresnel_s( i, n, method=0):
    """
    sin(i-r)    si cr - ci sr
    -------- =  -------------
    sin(i+r)    si cr + ci sr

    This form whilst pretty, gives nan at normal incidence, 0/0
    """
    si = np.sin(i)
    sr = si/n 

    if method == 0:
        ci = np.sqrt( 1 - si*si ) 
        cr = np.sqrt( 1 - sr*sr ) 
        num = si*cr - ci*sr
        den = si*cr + ci*sr 
    else:
        i = np.arcsin(si)
        r = np.arcsin(sr)
        num = np.sin(i - r)
        den = np.sin(i + r)
        #log.info("i %s r %s num %s den %s " % (i,r,num,den))
    pass
    return np.square(num/den)


def fresnel_p( i, n):
    """
    tan(i-r) 
    --------
    tan(i+r)   
    """
    si = np.sin(i)
    sr = si/n 
    i = np.arcsin(si)
    r = np.arcsin(sr)
    num = np.tan(i - r)
    den = np.tan(i + r)
    return np.square(num/den)



class Fresnel(object):
    def __init__(self, n1, n2, dom=None ):
        if dom is None:
            dom = np.linspace(0,90,91)

        n1 = np.asarray(n1)
        n2 = np.asarray(n2)

        th = dom*np.pi/180.
        spol = fresnel(th, n1, n2, True)
        ppol = fresnel(th, n1, n2, False)

        pass
        self.n1 = n1
        self.n2 = n2
        self.dom = dom
        self.th = th

        self.spol_0 = spol
        self.ppol_0 = ppol
        #self.alternative_check()

        self.cen  = (dom[:-1] + dom[1:])/2.
        # avg of bin edge values
        self.spol = (spol[:-1] + spol[1:])/2.  
        self.ppol = (ppol[:-1] + ppol[1:])/2.
        self.upol = (self.spol+self.ppol)/2.     # unpol?

        self.brewster = np.arctan(n2/n1)*180./np.pi
        self.critical = np.arcsin(n1/n2)*180./np.pi


    def alternative_check(self):
        """
        Alternative angle difference forms, misbehave at normal incidence
        Otherwise they match
        """
        th = self.th
        n1 = self.n1
        n2 = self.n2

        spol_0 = self.spol_0 
        ppol_0 = self.ppol_0 

        spol_2 = fresnel_s( th, n2/n1, method=1)
        spol_3 = fresnel_s( th, n2/n1, method=0)
        assert np.allclose( spol_0[1:], spol_2[1:] ), np.dstack([spol_0,spol_2, spol_3])
        assert np.allclose( spol_0[1:], spol_3[1:] ), np.dstack([spol_0,spol_2, spol_3])

        ppol_2 = fresnel_p( th, n2/n1)
        assert np.allclose( ppol_0[1:], ppol_2[1:] ), np.dstack([ppol_0, ppol_2])



    def __call__(self, xd, n):
        x = xd*np.pi/180.
        n1 = self.n1
        n2 = self.n2   
        cx = np.cos(x)
        sx = np.sin(x) 
        disc = 1. - np.square(n1*sx/n2)
        qdisc = np.sqrt(disc)
        pass
        spol = np.square((n1*cx - n2*qdisc)/(n1*cx + n2*qdisc)) 
        ppol = np.square((n1*qdisc - n2*cx)/(n1*qdisc + n2*cx)) 
        return n*(spol*f + (1.-f)*ppol)  


    def pl(self):
        plt.plot(self.cen, self.spol, label="S (perp)", c="r")
        plt.plot(self.cen, self.ppol, label="P (para)", c="b")

    def title(self):
        return "Fresnel %4.3f/%4.3f " % (self.n1, self.n2  )

    def plot(self, fig, ny=1, nx=1, n=1, log_=False):
        plt.title(self.title()) 
        ax = fig.add_subplot(ny,nx,n)
        self.pl()
        self.angles()
        legend = ax.legend(loc='upper left', shadow=True) 
        if log_:
            ax.set_yscale('log')


    def angles(self):
        a = self.brewster
        plt.plot([a, a], [1e-6, 1], 'k-', c="b", lw=2, label="Brewster")
        a = self.critical
        plt.plot([a, a], [1e-6, 1], 'k-', c="r", lw=2, label="Critical")

if __name__ == '__main__':


    n1 = np.array([1.])
    n2 = np.array([1.458])

    fr = Fresnel(n1,n2)
    fig = plt.figure()
    fr.plot(fig, log_=True) 
    fig.show()










