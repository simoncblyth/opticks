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
    return np.square(num/den) 





class Fresnel(object):
    def __init__(self, m1="Vacuum", m2="Pyrex", wavelength=500., dom=None ):
        if dom is None:
            dom = np.linspace(0,90,91)
        mlib = PropLib.PropLib("GMaterialLib")
        n1 = mlib.interp(m1,wavelength,PropLib.M_REFRACTIVE_INDEX)
        n2 = mlib.interp(m2,wavelength,PropLib.M_REFRACTIVE_INDEX)
        th = dom*np.pi/180.
        spol = fresnel(th, n1, n2, True)
        ppol = fresnel(th, n1, n2, False)
        pass
        self.mlib = mlib
        self.m1 = m1
        self.m2 = m2
        self.wl = wavelength
        self.n1 = n1
        self.n2 = n2
        self.dom = dom
        self.th = th
        self.spol = spol
        self.ppol = ppol
        self.upol = (spol+ppol)/2.   # unpol?

        # scrunch down to match bin counts
        self.a_spol = (spol[1:] + spol[:-1])/2.  
        self.a_ppol = (ppol[1:] + ppol[:-1])/2.
        self.brewster = np.arctan(n2/n1)*180./np.pi
        self.critical = np.arcsin(n1/n2)*180./np.pi

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
        plt.plot(self.dom, self.spol, label="S (perp)")
        plt.plot(self.dom, self.ppol, label="P (para)")

    def plot(self, fig, ny=1, nx=1, n=1):
        plt.title("Fresnel %s/%s %4.3f/%4.3f (%3d nm)" % (self.m1, self.m2, self.n1, self.n2, self.wl )) 
        ax = fig.add_subplot(ny,nx,n)
        self.pl()
        self.angles()
        legend = ax.legend(loc='upper left', shadow=True) 


    def angles(self):
        a = self.brewster
        plt.plot([a, a], [0, 1], 'k-', c="b", lw=2, label="Brewster")
        a = self.critical
        plt.plot([a, a], [0, 1], 'k-', c="r", lw=2, label="Critical")

if __name__ == '__main__':

    fr = Fresnel()
    fig = plt.figure()
    fr.plot(fig) 
    fig.show()










