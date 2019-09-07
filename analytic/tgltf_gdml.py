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

    run tgltf_gdml.py

    In [22]: tree.root.lv.solid.as_ncsg().transform
    [2017-05-19 13:25:49,429] p94731 {/Users/blyth/opticks/analytic/glm.py:175} WARNING - supressing identity transform

    In [29]: tree.get(1).lv.solid.as_ncsg()
    Out[29]: un(in(in(sp,sp),sp),cy)

    In [30]: cn = tree.get(1).lv.solid.as_ncsg()

    In [31]: cn.analyse()

    In [32]: cn.txt
    Out[32]: <opticks.analytic.textgrid.TextGrid at 0x10d01c450>

    In [33]: print cn.txt           union (of 3-sphere intersection and cylinder)
                         un    
                 in          cy
         in          sp        
     sp      sp                


Better way, use the prexisting::

    In [13]: sc.get_node(1).mesh
    Out[13]: Mh    0 :              pmt-hemi0xc0fed90 /dd/Geometry/PMT/lvPmtHemi0xc133740 

    In [14]: sc.get_node(1).mesh.csg
    Out[14]: un(in(in(sp,sp),sp),cy)height:3 totnodes:15 

    In [15]: print sc.get_node(1).mesh.csg.txt
                         un    
                 in          cy
         in          sp        
     sp      sp                





"""
import os, logging, sys, numpy as np

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.analytic.treebase import Tree, Node
from opticks.analytic.gdml import GDML
from opticks.analytic.gdml_builder import make_gdml, tostring_
from opticks.analytic.sc import Sc, Nd


if __name__ == '__main__':

    args = opticks_main()

    pmt = "/dd/Geometry/PMT/lvPmtHemi0xc133740"
    oil = "/dd/Geometry/AD/lvOIL0xbf5e0b8"

    skey = "pmt1"
#skey = "pmt2"
#skey = "pmt5"
#skey = "collar"
#skey = "collar2"

    gg = make_gdml(worldref=oil, structure_key=skey )
    wgg = GDML.wrap(gg) 

    tree = Tree(wgg.world)
    assert type(tree.root) is Node

    sc = Sc()
    sc.extras["verbosity"] = 3
    tg = sc.add_tree_gdml( tree.root, maxdepth=0 )
    assert type(tg) is Nd


    path = "$TMP/tgltf/%s.gltf" % (os.path.basename(sys.argv[0]))
    gltf = sc.save(path)
