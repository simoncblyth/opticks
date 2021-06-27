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
import os, numpy as np
from opticks.ana.nload import np_load
from opticks.ana.key import keydir
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)

    aa = np_load(os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy"))
    fc = np_load(os.path.join(kd,"GScintillatorLib/LS/FASTCOMPONENT.npy"))
    sc = np_load(os.path.join(kd,"GScintillatorLib/LS/SLOWCOMPONENT.npy"))
    a = aa[0,:,0]
    b = np.linspace(0,1,len(a))

    print("aa:%s" % str(aa.shape)) 
    print(" a:%s" % str(a.shape)) 
    print(" b:%s" % str(b.shape)) 
    print("fc:%s" % str(fc.shape)) 
    print("sc:%s" % str(sc.shape)) 

    assert aa.shape == (1, 4096, 1)
    assert np.all( fc == sc )
    assert a.shape == (4096,)

    fig = plt.figure()
    plt.title("Inverted Cumulative Distribution Function : for Scintillator Reemission " )
    ax = fig.add_subplot(1,1,1)
    ax.plot( b, a ) 
    ax.set_ylabel("Wavelength (nm)")
    ax.set_xlabel("Probability")
    fig.show()


    




