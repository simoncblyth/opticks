#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

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



::

    In [1]: np.dstack([w, a[0,:,0], a[0,:,1],a[0,:,2],a[0,:,3],a[1,:,0]])    ## Water
    ///
    ///                                     absorption     scattering
    ///                                     length (mm)    length
    ///
    ///                                     ~ 27m @ 440nm    1000m !!
    ///                                                     (placeholder value?)
    ///
    ///     water looks blue, 
    ///          more absorption (lower absorption lengths) at red high wavelength end
    ///          of the visible
    ///
    ///     scattering length 
    ///           1000m looks like a do-not scatter placeholder
    ///  
    ///
    Out[1]: 
    array([[[      60.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [      80.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     100.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     120.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     140.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     160.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     180.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     200.    ,        1.39  ,      691.5562,  1000000.    ,        0.    ,      300.    ],
            [     220.    ,        1.3841,     1507.1183,  1000000.    ,        0.    ,      300.    ],
            [     240.    ,        1.3783,     2228.2798,  1000000.    ,        0.    ,      300.    ],
            [     260.    ,        1.3724,     3164.6375,  1000000.    ,        0.    ,      300.    ],
            [     280.    ,        1.3666,     4286.0454,  1000000.    ,        0.    ,      300.    ],
            [     300.    ,        1.3608,     5992.6128,  1000000.    ,        0.    ,      300.    ],
            [     320.    ,        1.3595,     7703.5034,  1000000.    ,        0.    ,      300.    ],
            [     340.    ,        1.3585,    10257.2852,  1000000.    ,        0.    ,      300.    ],
            [     360.    ,        1.3572,    12811.0684,  1000000.    ,        0.    ,      300.    ],
            [     380.    ,        1.356 ,    15364.8496,  1000000.    ,        0.    ,      300.    ],
            [     400.    ,        1.355 ,    19848.9316,  1000000.    ,        0.    ,      300.    ],
            [     420.    ,        1.354 ,    24670.9512,  1000000.    ,        0.    ,      300.    ],
            [     440.    ,        1.353 ,    27599.9746,  1000000.    ,        0.    ,      300.    ],
            [     460.    ,        1.3518,    28732.2051,  1000000.    ,        0.    ,      300.    ],
            [     480.    ,        1.3505,    29587.0527,  1000000.    ,        0.    ,      300.    ],
            [     500.    ,        1.3492,    26096.2637,  1000000.    ,        0.    ,      300.    ],
            [     520.    ,        1.348 ,    17787.9492,  1000000.    ,        0.    ,      300.    ],
            [     540.    ,        1.347 ,    16509.3672,  1000000.    ,        0.    ,      300.    ],
            [     560.    ,        1.346 ,    13644.791 ,  1000000.    ,        0.    ,      300.    ],
            [     580.    ,        1.345 ,    10050.459 ,  1000000.    ,        0.    ,      300.    ],
            [     600.    ,        1.344 ,     4328.5166,  1000000.    ,        0.    ,      300.    ],
            [     620.    ,        1.3429,     3532.6135,  1000000.    ,        0.    ,      300.    ],
            [     640.    ,        1.3419,     3149.8655,  1000000.    ,        0.    ,      300.    ],
            [     660.    ,        1.3408,     2404.4004,  1000000.    ,        0.    ,      300.    ],
            [     680.    ,        1.3397,     2126.562 ,  1000000.    ,        0.    ,      300.    ],
            [     700.    ,        1.3387,     1590.72  ,  1000000.    ,        0.    ,      300.    ],
            [     720.    ,        1.3376,      809.6543,  1000000.    ,        0.    ,      300.    ],
            [     740.    ,        1.3365,      370.1322,  1000000.    ,        0.    ,      300.    ],
            [     760.    ,        1.3354,      371.9737,  1000000.    ,        0.    ,      300.    ],
            [     780.    ,        1.3344,      425.7059,  1000000.    ,        0.    ,      300.    ],
            [     800.    ,        1.3333,      486.681 ,  1000000.    ,        0.    ,      300.    ],
            [     820.    ,        1.3333,      486.681 ,  1000000.    ,        0.    ,      300.    ]]])


::

   ipython -i $(which material.py) -- --mat Water 
   ipython -i $(which material.py) -- --mat GdDopedLS



"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main
from opticks.ana.proplib import PropLib
   
mlib = PropLib("GMaterialLib")


class Material(object):
    def __init__(self, name):
        self.name = name
        self.mlib = mlib

    def lookup(self, prop, wavelength):
        return self.mlib.interp(self.name,wavelength,prop)

    def data(self):
        return self.mlib(self.name)

    def refractive_index(self, wavelength):
        return self.lookup(PropLib.M_REFRACTIVE_INDEX, wavelength)

    def absorption_length(self, wavelength):
        return self.lookup(PropLib.M_ABSORPTION_LENGTH, wavelength)

    def scattering_length(self, wavelength):
        return self.lookup(PropLib.M_SCATTERING_LENGTH, wavelength)
 
    def reemission_prob(self, wavelength):
        return self.lookup(PropLib.M_REEMISSION_PROB, wavelength)

    def group_velocity(self, wavelength):
        return self.lookup(PropLib.L_GROUP_VELOCITY, wavelength)

    def table(self, wl):
        """
        :param wl: array of wavelengths OR string "FINE_DOMAIN" or "COARSE_DOMAIN" to use standard sets
        :return tab: array of shape (num_wavelengths, 6) with wavelength and property values 
        """
        if type(wl) is str:
            if wl in "FINE_DOMAIN COARSE_DOMAIN".split():
                wl = getattr(PropLib, wl)
            pass  
        pass
        ri = self.refractive_index(wl)
        al = self.absorption_length(wl)
        sl = self.scattering_length(wl)
        rp = self.reemission_prob(wl)
        gv = self.group_velocity(wl)
        tab = np.dstack([wl,ri,al,sl,rp,gv])
        return tab 

    def table_qwn(self):
        qwn = "wavelen rindex abslen scatlen reemprob groupvel".split()
        return qwn 
 

    def dump(self):
        w = self.mlib.domain
        a = self.data()
        aa = np.dstack([w, a[0,:,0], a[0,:,1],a[0,:,2],a[0,:,3],a[1,:,0]])
        print("aa")
        print(aa)

    @classmethod
    def Hdr(cls):
        labels = "wavelen rindex abslen scatlen reemprob groupvel" 
        hdr = "   " + "".join(list(map(lambda _:"%12s" % _, labels.split())))
        return hdr 

    def hdr(self):
        return self.Hdr() + " " + self.name 



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    dmat = "Water"
    args = opticks_main(mat=dmat)    
    log.info("mat %s " % args.mat)

    wl = np.linspace(300.,600.,4)
    for name in list(map(lambda _:_.decode("utf-8"),mlib._names)):
        mat = Material(name)
        tab = mat.table(wl)
        print(mat.hdr())
        print(tab)
    pass        



if 0:
    import matplotlib.pyplot as plt 
    plt.ion()
    plt.plot( wl, al, "*", label="Absorption Length")
    plt.plot( wl, sl, "+", label="Scattering Length")
    plt.show()




