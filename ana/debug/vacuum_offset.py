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
Comparison of the old standard export with dpib shows the sagging vacuum
very clearly:: 

    In [4]: run vacuum_offset.py
    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    /usr/local/env/geant4/geometry/export/dpib/cfg4.dae
    -46.992 128.0
    -20.9888 128.0


"""
import os, sys, logging
import numpy as np
from opticks.ana.dae import DAE
import matplotlib.pyplot as plt
 
log = logging.getLogger(__name__)


def zr_plot(ax, a):
    r = np.linalg.norm(a[:,:2], 2, 1) * np.sign(a[:,0] )
    z = a[:,2]
    zhead = z[z>z.min()]
    print zhead.min(), zhead.max()
    ax.plot(z, r, "o")


if __name__ == '__main__':

    plt.close()
    plt.ion()

    path_0 = DAE.standardpath()
    path_1 = DAE.path("dpib", "cfg4.dae")

    print path_0
    print path_1

    dae_0 = DAE(path_0)
    a = dae_0.float_array("pmt-hemi-vac0xc21e248-Pos-array")

    dae_1 = DAE(path_1)
    b = dae_1.float_array("union-ab-i-6-fc-7-lc-110x7fd599d5e4b0-Pos-array")

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1, aspect='equal')
    zr_plot(ax, a )

    #ax = fig.add_subplot(2,1,2, aspect='equal')
    zr_plot(ax, b )



