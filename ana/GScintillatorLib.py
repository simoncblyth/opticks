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

   ipython -i $(which GScintillatorLib.py)

"""

from opticks.ana.nload import np_load
import matplotlib.pyplot as plt

if __name__ == '__main__':
    

    #f = np_load("$IDPATH/GScintillatorLib/LiquidScintillator/FASTCOMPONENT.npy")
    #print f  
    #plt.plot( f[:,0], f[:,1] ) 
    #plt.show()


    aa = np_load("$IDPATH/GScintillatorLib/GScintillatorLib.npy")

    assert aa.shape == (2, 4096, 1)
    assert np.all( aa[0] == aa[1] )

    a = aa[0,:,0]
    assert a.shape == (4096,)

    b = np.linspace(0,1,len(a))


    fig = plt.figure()

    plt.title("Inverted Cumulative Distribution Function : for Scintillator Reemission " )

    ax = fig.add_subplot(1,1,1)
    #ax.plot( a, b ) 
    ax.plot( b, a ) 
    ax.set_ylabel("Wavelength (nm)")
    ax.set_xlabel("Probability")

    fig.show()






     


