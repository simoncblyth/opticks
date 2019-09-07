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
https://matplotlib.org/gallery/shapes_and_collections/path_patch.html
"""
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


plt.ion()

fig, ax = plt.subplots()

Path = mpath.Path

"""

path_data = [
    (Path.MOVETO, (1.58, -2.57)),
    (Path.CURVE4, (0.35, -1.1)),
    (Path.CURVE4, (-1.75, 2.0)),
    (Path.CURVE4, (0.375, 2.0)),
    (Path.LINETO, (0.85, 1.15)),
    (Path.CURVE4, (2.2, 3.2)),
    (Path.CURVE4, (3, 0.05)),
    (Path.CURVE4, (2.0, -0.5)),
    (Path.CLOSEPOLY, (1.58, -2.57)),
    ]

"""

"""

       

       +---|---+
             r1
"""

r1 = 10
r2 = 20
hz = 5 

z2 =  hz 
z1 = -hz

path_data = [
   (Path.MOVETO,  ( -r1, z1)),
   (Path.LINETO,  ( -r2,  z2)),
   (Path.LINETO,  (  r2,  z2)),
   (Path.LINETO,  (  r1,  z1)),
   (Path.CLOSEPOLY, (-r1, z1)),
]

codes, verts = zip(*path_data)
path = mpath.Path(verts, codes)
patch = mpatches.PathPatch(path, fill=False)  # facecolor='r', alpha=0.5)
ax.add_patch(patch)

# plot control points and connecting lines
#x, y = zip(*path.vertices)
#line, = ax.plot(x, y, 'go-')

#ax.grid()
ax.axis('equal')
plt.show()
