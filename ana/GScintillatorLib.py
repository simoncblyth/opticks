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

   an ; ipython -i $(which GScintillatorLib.py)
   mkdir -p ~/simoncblyth.bitbucket.io/env/presentation/ana/GScintillatorLib
   cp /tmp/ana/GScintillatorLib/*.png ~/simoncblyth.bitbucket.io/env/presentation/ana/GScintillatorLib/

"""
import os, numpy as np
from opticks.ana.main import opticks_main
from opticks.ana.nload import np_load
from opticks.ana.key import keydir
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    ok = opticks_main()
    kd = keydir(os.environ["OPTICKS_KEY"])
    aa,aa_paths = np_load(os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy"))
    fc,fc_paths = np_load(os.path.join(kd,"GScintillatorLib/LS/FASTCOMPONENT.npy"))
    sc,sc_paths = np_load(os.path.join(kd,"GScintillatorLib/LS/SLOWCOMPONENT.npy"))
    a0 = aa[0,:,0]
    a1 = aa[1,:,0]
    a2 = aa[2,:,0]
    b = np.linspace(0,1,len(a0))

    print("aa:%s" % str(aa.shape)) 
    print("a0:%s" % str(a0.shape)) 
    print("a1:%s" % str(a1.shape)) 
    print("a2:%s" % str(a2.shape)) 
    print(" b:%s" % str(b.shape)) 
    print("fc:%s" % str(fc.shape)) 
    print("sc:%s" % str(sc.shape)) 

    assert aa.shape == (1, 4096, 1) or  aa.shape == (3, 4096, 1) 
    assert np.all( fc == sc )
    assert a0.shape == (4096,)
    assert a1.shape == (4096,)
    assert a2.shape == (4096,)

    fig = plt.figure(figsize=ok.figsize)
    plt.title("Inverted Cumulative Distribution Function for Scintillator Wavelength Generation " )
    ax = fig.add_subplot(1,1,1)
    ax.plot( b, a0, label="a0" ) 
    #ax.plot( b, a1, label="a1" ) 
    #ax.plot( b, a2, label="a2" ) 

    ax.set_ylabel("Wavelength (nm)")
    ax.set_xlabel("Probability")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plhs = 0.05
    prhs = 0.95

    wlhs = np.interp( plhs, b, a0 )
    wrhs = np.interp( prhs, b, a0 )

    ax.plot( [plhs,plhs], ylim, linestyle="dashed" )
    ax.plot( xlim       , [wlhs,wlhs], linestyle="dashed" )


    ax.plot( [prhs,prhs], ylim, linestyle="dashed" )
    ax.plot( xlim       , [wrhs,wrhs], linestyle="dashed" )

    fig.show()
    
    path="/tmp/ana/GScintillatorLib/icdf.png" 
    fold=os.path.dirname(path)
    if not os.path.isdir(fold):
       os.makedirs(fold)
    pass 
    log.info("save to %s " % path)
    fig.savefig(path)


   
