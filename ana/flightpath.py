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
flightpath.py
==============

FlightPath not in active use, easier to directly create the eye-look-up array, 
as done in mm0prim2.py  and use quiver plots to debug the path.


"""

import numpy as np, logging, os
log = logging.getLogger(__name__)
from opticks.ana.view import View
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



class FlightPath(object):
    """
    See optickscore/FlightPath.hh
    """
    FILENAME = "flightpath.npy"
    def __init__(self):
        self.views = []
  
    def as_array(self):
        log.info(" views %s " % len(self.views) )
        a = np.zeros( (len(self.views), 4, 4), dtype=np.float32 )
        for i, v in enumerate(self.views):
            a[i] = v.v
        pass
        return a 

    def save(self, dir_="/tmp"):
        path = os.path.join(dir_, self.FILENAME)
        a = fp.as_array()
        log.info("save %s to %s " % (repr(a.shape), path ))
        np.save(path, a ) 
        return a


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
    fp = FlightPath()

    dtype = np.float32
    f = np.linspace(0, 1, 10, dtype=dtype)[:-1]    # skip last to avoid repeating seam angle 
    t = f*2*np.pi 
    n = len(f)

    eye = np.zeros( [n,3], dtype=dtype)
    eye[:,0] = np.cos(t)  
    eye[:,1] = np.sin(t)  
    eye[:,2] = 2*f-1

    look = np.zeros( [n,3], dtype=dtype )
    look[:-1] = eye[1:]
    look[-1] = eye[0] 

    gaze = look - eye  

    up = np.zeros( [n,3], dtype=dtype )
    up[:] = [0,0,1] 

    v = np.zeros( (n,4,4), dtype=np.float32)
    v[:,0,:3] = eye
    v[:,1,:3] = look
    v[:,2,:3] = up

    np.save("/tmp/flightpath.npy", v )

    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    plt.title("flightpath")
    ax = fig.add_subplot(111, projection='3d')
    sz = 3 
    ax.set_ylim([-sz,sz])
    ax.set_xlim([-sz,sz])

    ax.plot( eye[:,0], eye[:,1], eye[:,2]  )
    fig.show()

"""
    for i in range(n):
        a = Arrow3D([eye[i,0], gaze[i,0]],
                    [eye[i,1], gaze[i,1]],
                    [eye[i,2], gaze[i,2]], mutation_scale=20,
                     lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    pass
"""



