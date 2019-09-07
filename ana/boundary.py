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
boundary.py : inner/outer materials and surfaces 
==================================================

Boundaries are composed of four parts:

* outer material
* outer surface, relevant to incoming photons
* inner surface, relevant to outgoing photons
* inner material

Boundaries are created from a specification string of form "omat/osur/isur/imat"
where outer and inner materials are required but surfaces are optional. 
For example:

.. code-block:: py

    In [2]: b1 = Boundary("Vacuum///GlassSchottF2")

    In [3]: wl = np.linspace(100.,730.,10, dtype=np.float32)

    In [4]: ri = b1.imat.refractive_index(wl)

    In [8]: al = b1.imat.absorption_length(wl)

    In [9]: sl = b1.imat.scattering_length(wl)

    In [10]: rp = b1.imat.reemission_prob(wl)

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
from opticks.ana.material import Material


class Boundary(object):
    def __init__(self, spec):
        self.spec = spec

        elem = spec.split("/")
        assert len(elem) == 4
        omat, osur, isur, imat = elem

        self.omat = Material(omat)
        self.osur = osur
        self.isur = isur
        self.imat = Material(imat)


    def title(self):
        return self.spec

    def __repr__(self):
        return "%s %s " % (  self.__class__.__name__ , self.spec )



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    opticks_environment()


    wl = np.linspace(100.,730.,10)

    boundary = Boundary("Vacuum///GlassSchottF2")

    print "imat",boundary.imat.refractive_index(wl)
    print "omat",boundary.omat.refractive_index(wl)




