#!/usr/bin/env python

import os, logging, sys, numpy as np

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.analytic.treebase import Tree
from opticks.analytic.gdml import GDML
from opticks.analytic.sc import Sc

args = opticks_main()
query = args.query 
log.info("%r" % query)

for i in np.arange(0, 12000, 500):
    print i, query(i)



if 0:

    oil = "/dd/Geometry/AD/lvOIL0xbf5e0b8"

    #sel = oil
    sel = 3153
    #sel = 1
    #sel = 0
    idx = 0 

    wgg = GDML.parse()
    tree = Tree(wgg.world)

    target = tree.findnode(sel=sel, idx=idx)



    sc = Sc(maxcsgheight=3)
    sc.extras["verbosity"] = 1
    sc.extras["targetnode"] = target.index

    tg = sc.add_tree_gdml( target, maxdepth=0)

    path = "$TMP/tgltf/$FUNCNAME.gltf"
    gltf = sc.save(path)

    print path      ## <-- WARNING COMMUNICATION PRINT

    #TODO: instead of just passing a path pass a config line or json snippet with the target

