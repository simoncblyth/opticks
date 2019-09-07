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

    In [3]: t0
    Out[3]: 
    array([[      1.    ,       0.    ,       0.    , ...,       0.    ,       0.    ,       1.    ],
           [     -0.5432,       0.8396,       0.    , ..., -802110.    ,   -2110.    ,       1.    ],
           [     -0.5432,       0.8396,       0.    , ..., -799739.375 ,    5390.    ,       1.    ],
           ..., 
           [      0.8396,       0.5432,       0.    , ..., -799312.625 ,   -7260.    ,       1.    ],
           [      0.2096,       0.9778,       0.    , ..., -794607.8125,   -7260.    ,       1.    ],
           [     -0.5432,       0.8396,       0.    , ..., -802110.    ,  -12410.    ,       1.    ]], dtype=float32)

    In [4]: t0.shape
    Out[4]: (12230, 16)

    In [6]: print t0[3159].reshape(4,4)
    [[      0.5432      -0.8396       0.           0.    ]
     [      0.8396       0.5432       0.           0.    ]
     [      0.           0.           1.           0.    ]
     [ -18079.4531 -799699.4375   -7100.           1.    ]]


imon:GMergedMesh blyth$ cd 0
simon:0 blyth$ l
total 39992
-rw-r--r--  1 blyth  staff       96 Jun 14 16:21 aiidentity.npy
-rw-r--r--  1 blyth  staff   293600 Jun 14 16:21 bbox.npy
-rw-r--r--  1 blyth  staff  1739344 Jun 14 16:21 boundaries.npy
-rw-r--r--  1 blyth  staff   195760 Jun 14 16:21 center_extent.npy
-rw-r--r--  1 blyth  staff  2702480 Jun 14 16:21 colors.npy
-rw-r--r--  1 blyth  staff   195760 Jun 14 16:21 identity.npy
-rw-r--r--  1 blyth  staff   195760 Jun 14 16:21 iidentity.npy
-rw-r--r--  1 blyth  staff  5217872 Jun 14 16:21 indices.npy
-rw-r--r--  1 blyth  staff      144 Jun 14 16:21 itransforms.npy
-rw-r--r--  1 blyth  staff    49000 Jun 14 16:21 meshes.npy
-rw-r--r--  1 blyth  staff   195760 Jun 14 16:21 nodeinfo.npy
-rw-r--r--  1 blyth  staff  1739344 Jun 14 16:21 nodes.npy
-rw-r--r--  1 blyth  staff  2702480 Jun 14 16:21 normals.npy
-rw-r--r--  1 blyth  staff  1739344 Jun 14 16:21 sensors.npy
-rw-r--r--  1 blyth  staff   782800 Jun 14 16:21 transforms.npy
-rw-r--r--  1 blyth  staff  2702480 Jun 14 16:21 vertices.npy
simon:0 blyth$ 


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.base import opticks_main, idp_


if __name__ == '__main__':
     tr = np.load(idp_("GMergedMesh/0/transforms.npy"))
     bb = np.load(idp_("GMergedMesh/0/bbox.npy"))             # 
     ce = np.load(idp_("GMergedMesh/0/center_extent.npy"))    # (12230, 4)
     ms = np.load(idp_("GMergedMesh/0/meshes.npy"))           # mesh index (12230, 1)
     vv = np.load(idp_("GMergedMesh/0/vertices.npy"))         #  (225200, 3) ... global positions
     nn = np.load(idp_("GMergedMesh/0/nodes.npy"))            #  (434816, 1)    nn.min() = 3153   nn.max() = 12220   triface->nodeidx
     ni = np.load(idp_("GMergedMesh/0/nodeinfo.npy"))         #  (12230, 4)    ni[:,0].sum() = 434816   ni[:,1].sum() = 225200   nf/nv/nix/pix








