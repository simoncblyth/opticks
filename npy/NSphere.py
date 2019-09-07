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


import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.rcParams['legend.fontsize'] = 10




class NSphere(object):
    def __init__(self, radius, ndiv):
        self.radius = radius 
        self.ndiv = ndiv

        azimuthal = np.linspace(0, 2*np.pi, ndiv )
        polar = np.linspace(0, np.pi, ndiv )
        ca = np.cos(azimuthal)
        sa = np.sin(azimuthal)
        cp = np.cos(polar)
        sp = np.sin(polar)

        self.azimuthal = azimuthal
        self.polar = polar

        self.ca = ca
        self.sa = sa
        self.cp = cp
        self.sp = sp
    
    def xyz(self, i):
        cp = np.repeat( self.cp[i], self.ndiv )
        sp = np.repeat( self.sp[i], self.ndiv )

        x = sp*self.ca*self.radius
        y = sp*self.sa*self.radius
        z = cp*self.radius

        return x,y,z



if __name__ ==  '__main__':


    ndiv = 5
    sph = NSphere(10,ndiv)

    plt.ion()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(ndiv):
        x,y,z = sph.xyz(i)
        ax.plot(x,y,z)
    pass

    plt.show()







