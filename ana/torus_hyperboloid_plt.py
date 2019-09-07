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
Continuing from tboolean-12

TODO: merge this with the much better plotting technique (deferred placement) of x018_torus_hyperboloid_plt.py 

"""

import numpy as np, math 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
import matplotlib.lines as mlines

from opticks.ana.torus_hyperboloid import Tor, Hyp


def make_rect( cxy , wh, **kwa ):
    """
    :param cxy: center of rectangle
    :param wh: width, height
    """
    ll = ( cxy[0] - wh[0]/2., cxy[1] - wh[1]/2. )
    return Rectangle( ll,  wh[0], wh[1], **kwa  ) 


if __name__ == '__main__':


    R,r = 97.000,52.010
    ch,cz,cn = 23.783,-23.773,-195.227
    cyr = 75.951

    r0 = R - r 
    rr0 = r0*r0

    tor = Tor(R,r)
    assert tor.rz(0) == R - r 
    assert tor.rz(r) == R  

    # in torus/hyp frame cylinder top and bottom at

    ztop, zbot = ch - cz, -ch - cz  #     (47.556, -0.010000000000001563)
    rtop, rbot = tor.rz(ztop), tor.rz(zbot)

    zf = Hyp.ZF( rbot, ztop, rtop )
    hyp = Hyp( rbot, zf )


    #sz = R+1.5*r
    sz = 400 


    exy,ez = 1.391,1.000
    era = 179.00


    bulb = Ellipse( xy=(0,0), width=2*exy*era, height=2*ez*era, fill=False )  


    rhs = Circle( (R,cz),  radius=r, fill=False) 
    lhs = Circle( (-R,cz),  radius=r, fill=False) 

    cy = make_rect( (0,0), (2*cyr,2*ch), fill=False )

    byr = 45.010
    byh = 57.510
    cybase = make_rect( (0,-276.500), (2*byr, 2*byh), fill=False ) 

    cur = 254.00
    cuh = 92.000

    cycut = make_rect( (0,cuh) ,  (2*cur, 2*cuh), fill=False )


    plt.ion()
    fig = plt.figure(figsize=(5,5))
    plt.title("torus_hyperboloid_plt")

    ax = fig.add_subplot(111)
    ax.set_ylim([-sz,sz])
    ax.set_xlim([-sz,sz])

    ax.add_patch( bulb )
    ax.add_patch( lhs )
    ax.add_patch( rhs )
    ax.add_patch( cy )
    ax.add_patch( cybase )
    ax.add_patch( cycut )

    z = np.linspace( -sz, sz, 100 )

    dz = cz
    rz = hyp.rz(z) 

    ax.plot( rz, z + dz, c="b") 
    ax.plot( -rz, z + dz, c="b") 

    
    fig.show()




