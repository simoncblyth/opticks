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
    def __init__(self, boundary, wavelength=500., dom=None ):
        if dom is None:
            dom = np.linspace(0,90,91)

        n1 = boundary.omat.refractive_index(wavelength)  
        n2 = boundary.imat.refractive_index(wavelength)  

        th = dom*np.pi/180.
        spol = fresnel(th, n1, n2, True)
        ppol = fresnel(th, n1, n2, False)
        pass
        self.boundary = boundary 

        self.wl = wavelength
        self.n1 = n1
        self.n2 = n2
        self.dom = dom
        self.th = th

        self.cen  = (dom[:-1] + dom[1:])/2.
        # avg of bin edge values
        self.spol = (spol[:-1] + spol[1:])/2.  
        self.ppol = (ppol[:-1] + ppol[1:])/2.
        self.upol = (self.spol+self.ppol)/2.     # unpol?

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
        plt.plot(self.cen, self.spol, label="S (perp)", c="r")
        plt.plot(self.cen, self.ppol, label="P (para)", c="b")

    def title(self):
        return "Fresnel %s %4.3f/%4.3f (%3d nm)" % (self.boundary.title(), self.n1, self.n2, self.wl )

    def plot(self, fig, ny=1, nx=1, n=1):
        plt.title(self.title()) 
        ax = fig.add_subplot(ny,nx,n)
        self.pl()
        self.angles()
        legend = ax.legend(loc='upper left', shadow=True) 


    def angles(self):
        a = self.brewster
        plt.plot([a, a], [1e-6, 1], 'k-', c="b", lw=2, label="Brewster")
        a = self.critical
        plt.plot([a, a], [1e-6, 1], 'k-', c="r", lw=2, label="Critical")

if __name__ == '__main__':

    fr = Fresnel()
    fig = plt.figure()
    fr.plot(fig) 
    fig.show()










