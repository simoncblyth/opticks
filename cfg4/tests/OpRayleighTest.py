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
::

    OpRayleighTest   # cfg4-
    ORayleighTest    # oxrap-

    ipython -i OpRayleighTest.py 

"""

X,Y,Z=0,1,2
OLDMOM,OLDPOL,NEWMOM,NEWPOL = 0,1,2,3

import os, numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    aa = np.load(os.path.expandvars("$TMP/RayleighTest/ok.npy"))
    bb = np.load(os.path.expandvars("$TMP/RayleighTest/cfg4.npy"))

    bins = 100 
    nx = 6
    ny = 2

    x0 = aa[:,NEWMOM,X]
    y0 = aa[:,NEWMOM,Y]
    z0 = aa[:,NEWMOM,Z]

    a0 = aa[:,NEWPOL,X]
    b0 = aa[:,NEWPOL,Y]
    c0 = aa[:,NEWPOL,Z]

    x1 = bb[:,NEWMOM,X]
    y1 = bb[:,NEWMOM,Y]
    z1 = bb[:,NEWMOM,Z]

    a1 = bb[:,NEWPOL,X]
    b1 = bb[:,NEWPOL,Y]
    c1 = bb[:,NEWPOL,Z]

    qwns = [
             (1,x0),(2,y0),(3,z0),(4,a0),(5,b0),(6,c0),
             (7,x1),(8,y1),(9,z1),(10,a1),(11,b1),(12,c1)
          ]

    for i,q in qwns:
        plt.subplot(ny, nx, i)
        plt.hist(q, bins=bins)
    pass
    plt.show()


    #dyz, ye, ze = np.histogram2d(y, z, bins=(100,100))
    #extent = [ye[0], ye[-1], ze[0], ze[-1]]
    #plt.imshow(dyz.T, extent=extent, origin='lower')
    #plt.show()


