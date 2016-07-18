#!/usr/bin/env python
"""
material.py : optical properties access
==================================================

Geocache materials are accessible by name, usage example: 

.. code-block:: py

    In [2]: mat = Material("GlassSchottF2")

    In [3]: wl = np.linspace(100.,730.,10, dtype=np.float32)

    In [4]: ri = mat.refractive_index(wl)

    In [8]: al = mat.absorption_length(wl)

    In [9]: sl = mat.scattering_length(wl)

    In [10]: rp = mat.reemission_prob(wl)

    In [11]: np.dstack([wl,ri,al,sl,rp])
    Out[11]: 
    array([[[     100.   ,        1.685,  1000000.   ,  1000000.   ,        0.   ],
            [     170.   ,        1.685,  1000000.   ,  1000000.   ,        0.   ],
            [     240.   ,        1.685,  1000000.   ,  1000000.   ,        0.   ],
            [     310.   ,        1.685,  1000000.   ,  1000000.   ,        0.   ],
            [     380.   ,        1.658,  1000000.   ,  1000000.   ,        0.   ],
            [     450.   ,        1.638,  1000000.   ,  1000000.   ,        0.   ],
            [     520.   ,        1.626,  1000000.   ,  1000000.   ,        0.   ],
            [     590.   ,        1.619,  1000000.   ,  1000000.   ,        0.   ],
            [     660.   ,        1.614,  1000000.   ,  1000000.   ,        0.   ],
            [     730.   ,        1.611,  1000000.   ,  1000000.   ,        0.   ]]])


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_environment
from opticks.ana.proplib import PropLib


class Material(object):
    def __init__(self, name):
        self.name = name
        self.mlib = None

    def lookup(self, prop, wavelength):
        if self.mlib is None:
            self.mlib = PropLib("GMaterialLib")
        pass
        return self.mlib.interp(self.name,wavelength,prop)
 
    def refractive_index(self, wavelength):
        return self.lookup(PropLib.M_REFRACTIVE_INDEX, wavelength)

    def absorption_length(self, wavelength):
        return self.lookup(PropLib.M_ABSORPTION_LENGTH, wavelength)

    def scattering_length(self, wavelength):
        return self.lookup(PropLib.M_SCATTERING_LENGTH, wavelength)
 
    def reemission_prob(self, wavelength):
        return self.lookup(PropLib.M_REEMISSION_PROB, wavelength)





if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    mat = Material("GlassSchottF2")

    wl = np.linspace(100.,730.,10)

    ri = mat.refractive_index(wl)

    al = mat.absorption_length(wl)

    sl = mat.scattering_length(wl)

    rp = mat.reemission_prob(wl)

    print np.dstack([wl,ri,al,sl,rp])
 

