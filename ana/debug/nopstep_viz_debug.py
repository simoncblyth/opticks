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
nopstep_viz_debug.py: Creates fake nopstep (non-photon step) for visualization debugging
============================================================================================

See okc/OpticksEvent::setFakeNopstepPath


"""
import os, logging
import numpy as np

import matplotlib.pyplot as plt

from opticks.ana.base import opticks_environment
from opticks.ana.nload import A
from opticks.ana.debug.CGDMLDetector import CGDMLDetector

def make_fake_nopstep(tk):
    """
    :param tk: dict of dict containing track parameters
    :return nop:  numpy array of shape (tot n, 4, 4)
                  containing time and global positions in [:,0] 

    For each track

    *n* 
          number of time steps 
    *tmin*
          ns
    *tmax*
          ns
    *fn*
          parametric equation returning local coordinate [0,1,2] from time input           
    *frame*
          4x4 homogenous matrix applied to the local trajectory coordinates
          to get global coordinates

    """
    traj = {}
    ftraj = {}

    for k in tk.keys():

        tkd = tk[k]
        n = tkd["n"]
        fn = tkd["fn"]
        mat = tkd["frame"]

        traj[k] = np.ones([n, 4], dtype=np.float32)
        t = np.linspace(tkd["tmin"],tkd["tmax"], n)

        for p in range(n):
            traj[k][p,:3] = fn(t[p])    
        pass

        ftraj[k] = np.dot(traj[k], mat)   
        ftraj[k][:,3] = t 
    pass 

    combi = np.vstack(ftraj.values())
    nop = np.zeros([combi.shape[0], 4, 4], dtype=np.float32)
    nop[:,0] = combi 

    return nop



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    frame = 3153

    det = CGDMLDetector()
    mat = det.getGlobalTransform(frame)
    print "mat %s " % repr(mat)

    tk = {} 
    tk["+x"] = dict(n=10,frame=mat,tmin=0,tmax=10,fn=lambda t:np.array([0,0,0]) + t*np.array([100,0,0]))
    tk["-x"] = dict(n=20,frame=mat,tmin=1,tmax=20,fn=lambda t:np.array([0,0,0]) + t*np.array([-100,0,0]))
    tk["+y"] = dict(n=10,frame=mat,tmin=0,tmax=10,fn=lambda t:np.array([0,0,0]) + t*np.array([0,100,0]))
    tk["-y"] = dict(n=20,frame=mat,tmin=1,tmax=20,fn=lambda t:np.array([0,0,0]) + t*np.array([0,-100,0]))
    tk["+z"] = dict(n=10,frame=mat,tmin=0,tmax=10,fn=lambda t:np.array([0,0,0]) + t*np.array([0,0,100]))
    tk["-z"] = dict(n=20,frame=mat,tmin=1,tmax=20,fn=lambda t:np.array([0,0,0]) + t*np.array([0,0,-100]))


    nop = make_fake_nopstep(tk)

    path = "$TMP/fake_nopstep.npy"
    np.save(os.path.expandvars(path), nop)
 
    a = np.load(os.path.expandvars(path))


