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

    In [23]: txf[:,3,:4]
    Out[23]: 
    array([[[    0.    ,     0.    ,     1.    ,     0.    ],
            [   -0.7934,     0.6088,     0.    ,     0.    ],
            [   -0.6088,    -0.7934,     0.    ,     0.    ],
            [-1415.0659, -1844.1498, -1750.    ,     1.    ]],

           [[    0.    ,     0.    ,     1.    ,     0.    ],
            [   -0.7934,     0.6088,     0.    ,     0.    ],
            [   -0.6088,    -0.7934,     0.    ,     0.    ],
            [-1415.0659, -1844.1498, -1250.    ,     1.    ]],

           [[    0.    ,     0.    ,     1.    ,     0.    ],
            [   -0.7934,     0.6088,     0.    ,     0.    ],
            [   -0.6088,    -0.7934,     0.    ,     0.    ],
            [-1415.0659, -1844.1498,  -750.    ,     1.    ]],

           [[    0.    ,     0.    ,     1.    ,     0.    ],
            [   -0.7934,     0.6088,     0.    ,     0.    ],
            [   -0.6088,    -0.7934,     0.    ,     0.    ],
            [-1415.0659, -1844.1498,  -250.    ,     1.    ]],

           [[    0.    ,     0.    ,     1.    ,     0.    ],
            [   -0.7934,     0.6088,     0.    ,     0.    ],
            [   -0.6088,    -0.7934,     0.    ,     0.    ],
            [-1415.0659, -1844.1498,   250.    ,     1.    ]],

           [[    0.    ,     0.    ,     1.    ,     0.    ],
            [   -0.7934,     0.6088,     0.    ,     0.    ],
            [   -0.6088,    -0.7934,     0.    ,     0.    ],
            [-1415.0659, -1844.1498,   750.    ,     1.    ]],

           [[    0.    ,     0.    ,     1.    ,     0.    ],
            [   -0.7934,     0.6088,     0.    ,     0.    ],
            [   -0.6088,    -0.7934,     0.    ,     0.    ],
            [-1415.0659, -1844.1498,  1250.    ,     1.    ]],

           [[    0.    ,     0.    ,     1.    ,     0.    ],
            [   -0.7934,     0.6088,     0.    ,     0.    ],
            [   -0.6088,    -0.7934,     0.    ,     0.    ],
            [-1415.0659, -1844.1498,  1750.    ,     1.    ]]], dtype=float32)



"""
import numpy as np
import os, logging
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.pmt.gdml import GDML
from opticks.ana.pmt.treebase import Tree
from opticks.dev.csg.sc import Sc


def dump_tree_transforms(nodes):
    for i,node in enumerate(nodes):
        print "%3d %50s %s " % ( i, repr(node.pv.rotation), repr(node.pv.position)  )
        #print node.pv.position 
        #print node.pv.rotation
        #print node.pv.transform


if __name__ == '__main__':

    args = opticks_main()

    gsel = args.gsel            # string representing target node index integer or lvname
    gmaxnode = args.gmaxnode    # limit subtree node count
    gmaxdepth = args.gmaxdepth  # limit subtree node depth from the target node
    gidx = args.gidx            # target selection index, used when the gsel-ection yields multiple nodes eg when using lvname selection 

    gsel = "/dd/Geometry/AD/lvSST0x" 
    gmaxdepth = 3

    log.info(" gsel:%s gidx:%s gmaxnode:%s gmaxdepth:%s " % (gsel, gidx, gmaxnode, gmaxdepth))

    gdmlpath = os.environ['OPTICKS_GDMLPATH']   # set within opticks_main 
    gdml = GDML.parse(gdmlpath)


    tree = Tree(gdml.world)
    target = tree.findnode(gsel, gidx)

    log.info(" target node %s " % target )   


    pmtnodes = target.find_nodes_lvn("/dd/Geometry/PMT/lvPmtHemi0x")  # treebase nodes, closer to GDML source than Nd nodes
    assert len(pmtnodes) == 192    

    #dump_tree_transforms(pmtnodes)



if 1:
    sc = Sc()
    tg = sc.add_tree_gdml( target, maxdepth=3 )
    gltf = sc.save("$TMP/nd/scene.gltf")

    lvIdx = sc.find_meshes_so("pmt-hemi")[0].lvIdx    
    pmts = tg.find_nodes(lvIdx)      # Sc/Nd nodes
    assert len(pmts) == 192 == 24*8
    txf = np.vstack([pmt.transform for pmt in pmts]).reshape(8,24,4,4)





