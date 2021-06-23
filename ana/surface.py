#!/usr/bin/env python
"""
surface.py : optical properties access
==================================================

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main
from opticks.ana.proplib import PropLib
   
slib = PropLib("GSurfaceLib")


class Surface(object):
    def __init__(self, name):
        self.name = name
        self.slib = slib

    def lookup(self, prop, wavelength):
        return self.slib.interp(self.name,wavelength,prop)

    def data(self):
        return self.slib(self.name)

    def detect(self, wavelength):
        return self.lookup(PropLib.S_DETECT, wavelength)

    def absorb(self, wavelength):
        return self.lookup(PropLib.S_ABSORB, wavelength)

    def rspecular(self, wavelength):
        return self.lookup(PropLib.S_REFLECT_SPECULAR, wavelength)
 
    def rdiffuse(self, wavelength):
        return self.lookup(PropLib.S_REFLECT_DIFFUSE, wavelength)

    def table(self, wl):
        sd = self.detect(wl)
        sa = self.absorb(wl)
        sr = self.rspecular(wl)
        dr = self.rdiffuse(wl)
        tab = np.dstack([wl,sd,sa,sr,dr])
        return tab 

    def dump(self):
        w = self.mlib.domain
        a = self.data()
        aa = np.dstack([w, a[0,:,0], a[0,:,1],a[0,:,2],a[0,:,3],a[1,:,0]])
        print("aa")
        print(aa)

    @classmethod
    def Hdr(cls):
        labels = "wl sd sa sr dr" 
        hdr = "".join(list(map(lambda _:"%8s" % _, labels.split())))
        return hdr 

    def hdr(self):
        return self.Hdr() + " " + self.name 



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    args = opticks_main()    

    wl = np.linspace(300.,600.,4)
    for name in slib._names:
        surf = Surface(name)
        tab = surf.table(wl)
        print(surf.hdr())
        print(tab)
    pass        


if 0:
    import matplotlib.pyplot as plt 
    plt.ion()
    plt.plot( wl, al, "*", label="Absorption Length")
    plt.plot( wl, sl, "+", label="Scattering Length")
    plt.show()




